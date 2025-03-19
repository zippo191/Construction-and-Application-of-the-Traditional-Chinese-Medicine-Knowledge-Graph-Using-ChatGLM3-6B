import os
import json
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, random_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# ====== transformers组件 ======
from transformers import (
    BertTokenizerFast,
    BertModel,
    BertForTokenClassification,
    RobertaForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)

try:
    from torchcrf import CRF
except ImportError:
    raise ImportError("Please install torchcrf via: pip install torchcrf")


class TCMEntityDataset(Dataset):


    def __init__(self, data_list, tokenizer, max_len=128):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        text = list(item["text"])  # 将句子转为字符列表
        labels = ["O"] * len(text)

        for ent in item.get("entities", []):
            etype = ent["entity"]  # "药材" / "症状"
            start, end = ent["start"], ent["end"]
            if etype in ("药材", "症状"):
                # 确保不越界
                if 0 <= start < len(text) and 0 < end <= len(text):
                    labels[start] = f"B-{etype}"
                    for i in range(start+1, end):
                        labels[i] = f"I-{etype}"

        # 分词
        encoding = self.tokenizer(
            text,
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len
        )

        # 同步BIO标签
        labels = labels[:self.max_len] + ["O"] * (self.max_len - len(labels))
        label_ids = [LABEL2ID.get(l, LABEL2ID["O"]) for l in labels]
        encoding["labels"] = label_ids

        return {k: torch.tensor(v) for k, v in encoding.items()}


LABEL2ID = {
    "O": 0,
    "B-药材": 1,
    "I-药材": 2,
    "B-症状": 3,
    "I-症状": 4
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}



class EnhancedTCMER(nn.Module):
    """
    改进版TCMER: BERT + 双层BiLSTM + 多卷积核CNN + MultiheadSelfAttn + CRF
    """
    def __init__(
        self,
        bert_name="bert_base_chinese",
        num_labels=5,
        lstm_hidden=768,
        num_lstm_layers=2,
        cnn_channels=256,
        dropout_prob=0.3
    ):
        super().__init__()
        self.num_labels = num_labels

        # 本地BERT
        self.bert = BertModel.from_pretrained(
            bert_name,
            local_files_only=True
        )
        hidden_size = self.bert.config.hidden_size

        # 双层双向LSTM + LayerNorm
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden // 2,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        self.lstm_ln = nn.LayerNorm(lstm_hidden)

        # 多卷积核 CNN
        self.kernel_sizes = [3, 5, 7]
        self.conv_list = nn.ModuleList([
            nn.Conv1d(
                in_channels=lstm_hidden,
                out_channels=cnn_channels,
                kernel_size=k,
                padding=k // 2
            )
            for k in self.kernel_sizes
        ])
        total_cnn_out = cnn_channels * len(self.kernel_sizes)

        # Multi-head Self-Attention
        self.attn = nn.MultiheadAttention(embed_dim=total_cnn_out, num_heads=4, batch_first=True)
        self.attn_ln = nn.LayerNorm(total_cnn_out)

        self.dropout = nn.Dropout(dropout_prob)

        # 输出层 + CRF
        self.hidden2tag = nn.Linear(total_cnn_out, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # BERT
        bert_out = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        seq_out = bert_out.last_hidden_state  # [batch, seq_len, hidden_size]

        # LSTM
        lstm_out, _ = self.lstm(seq_out)
        lstm_out = self.lstm_ln(lstm_out)

        # 多卷积核
        lstm_out_t = lstm_out.transpose(1, 2)
        conv_feats = []
        for conv in self.conv_list:
            c = conv(lstm_out_t)
            c = torch.relu(c)
            conv_feats.append(c.transpose(1, 2))
        cnn_cat = torch.cat(conv_feats, dim=2)  # [batch, seq_len, total_cnn_out]

        # Self-Attn + 残差
        attn_out, _ = self.attn(cnn_cat, cnn_cat, cnn_cat)
        attn_out = self.attn_ln(attn_out + cnn_cat)
        attn_out = self.dropout(attn_out)

        # 输出 + CRF
        logits = self.hidden2tag(attn_out)
        loss = None
        if labels is not None:
            mask = attention_mask.bool() if attention_mask is not None else None
            log_likelihood = self.crf(logits, labels, mask=mask, reduction="mean")
            loss = -log_likelihood

        return {"loss": loss, "logits": logits}


# 其他两个模型：纯Transformer-CRF(Bert), RoBERTa-CRF
class BertCRFForTokenClassification(nn.Module):
    def __init__(self, bert_name="bert_base_chinese", num_labels=5):
        super().__init__()
        self.num_labels = num_labels
        self.bertfc = BertForTokenClassification.from_pretrained(
            bert_name,
            num_labels=num_labels,
            local_files_only=True
        )
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bertfc.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        seq_out = outputs.last_hidden_state
        logits = self.bertfc.dropout(seq_out)
        logits = self.bertfc.classifier(logits)

        loss = None
        if labels is not None:
            mask = attention_mask.bool()
            log_likelihood = self.crf(logits, labels, mask=mask, reduction="mean")
            loss = -log_likelihood
        return {"loss": loss, "logits": logits}


class RobertaCRFForTokenClassification(nn.Module):
    def __init__(self, roberta_name="roberta_base_chinese", num_labels=5):
        super().__init__()
        self.num_labels = num_labels
        self.robertafc = RobertaForTokenClassification.from_pretrained(
            roberta_name,
            num_labels=num_labels,
            local_files_only=True
        )
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.robertafc.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        seq_out = outputs.last_hidden_state
        logits = self.robertafc.dropout(seq_out)
        logits = self.robertafc.classifier(logits)

        loss = None
        if labels is not None:
            mask = attention_mask.bool()
            log_likelihood = self.crf(logits, labels, mask=mask, reduction="mean")
            loss = -log_likelihood
        return {"loss": loss, "logits": logits}


from transformers import Trainer, DataCollatorForTokenClassification, TrainingArguments

def compute_metrics_fn(pred):
    predictions = np.argmax(pred.predictions, axis=2)
    labels = pred.label_ids
    true_labels, pred_labels = [], []
    for i, label_seq in enumerate(labels):
        for j, lb_id in enumerate(label_seq):
            if lb_id != -100:
                true_labels.append(ID2LABEL[lb_id])
                pred_labels.append(ID2LABEL[predictions[i][j]])

    report = classification_report(true_labels, pred_labels, output_dict=True)
    macro_p = report["macro avg"]["precision"]
    macro_r = report["macro avg"]["recall"]
    macro_f1 = report["macro avg"]["f1-score"]
    return {
        "precision": macro_p,
        "recall": macro_r,
        "f1": macro_f1,
        "report_dict": report
    }

def save_classification_report(report_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    import json
    # 保存详细分类指标
    save_path = os.path.join(output_dir, "classification_report.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)

    # 写入简要结果
    metrics_txt = os.path.join(output_dir, "metrics.txt")
    with open(metrics_txt, "w", encoding="utf-8") as f:
        macro_f1 = report_dict["macro avg"]["f1-score"]
        macro_p = report_dict["macro avg"]["precision"]
        macro_r = report_dict["macro avg"]["recall"]
        f.write(f"Macro-F1: {macro_f1:.4f}\n")
        f.write(f"Macro-Precision: {macro_p:.4f}\n")
        f.write(f"Macro-Recall: {macro_r:.4f}\n")

    # 画柱状图
    labels, p_vals, r_vals, f_vals = [], [], [], []
    for k, v in report_dict.items():
        if k in ["accuracy", "macro avg", "weighted avg"]:
            continue
        labels.append(k)
        p_vals.append(v["precision"])
        r_vals.append(v["recall"])
        f_vals.append(v["f1-score"])

    x = np.arange(len(labels))
    width = 0.2
    plt.figure(figsize=(10,6))
    plt.bar(x - width, p_vals, width=width, label="Precision")
    plt.bar(x, r_vals, width=width, label="Recall")
    plt.bar(x + width, f_vals, width=width, label="F1")
    plt.xticks(x, labels, rotation=45)
    plt.ylim(0, 1.05)
    plt.title("Class-wise metrics (Precision, Recall, F1) [English Caption]")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "class_score_bar.png")
    plt.savefig(fig_path, dpi=120)
    plt.close()

def plot_loss_curve(log_history, output_dir):
    train_loss, eval_loss, epochs = [], [], []
    for rec in log_history:
        if "loss" in rec and "epoch" in rec:
            train_loss.append(rec["loss"])
            epochs.append(rec["epoch"])
        if "eval_loss" in rec and "epoch" in rec:
            eval_loss.append((rec["epoch"], rec["eval_loss"]))

    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_loss, marker="o", label="Train Loss")
    eval_ep = [ep for ep, _ in eval_loss]
    eval_val = [val for _, val in eval_loss]
    plt.plot(eval_ep, eval_val, marker="o", label="Eval Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Evaluation Loss Curve")
    plt.legend()
    plt.tight_layout()
    curve_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(curve_path, dpi=120)
    plt.close()

def train_and_evaluate(model, train_ds, eval_ds, model_name="MyModel"):
    data_collator = DataCollatorForTokenClassification(tokenizer_bert)
    output_dir = f"./results_{model_name}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=10,
        logging_dir=f"./logs_{model_name}",
        disable_tqdm=False,
        learning_rate=2e-5
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn
    )

    # 训练
    trainer.train()
    # 最后一轮评估
    eval_metrics = trainer.evaluate()
    print(f"[{model_name}] Final Eval (macro): {eval_metrics}")

    # 获取详细report
    pred = trainer.predict(eval_ds)
    pred_report = compute_metrics_fn(pred)["report_dict"]

    # 保存结果 & 作图
    save_classification_report(pred_report, output_dir)
    plot_loss_curve(trainer.state.log_history, output_dir)

    return trainer


if __name__ == "__main__":
    # 检查本地文件
    if not os.path.exists("tcm_data_clean.json"):
        print("Error: tcm_data_clean.json not found!")
    if not os.path.isdir("bert_base_chinese"):
        print("Warning: bert_base_chinese/ not found!")
    if not os.path.isdir("roberta_base_chinese"):
        print("Warning: roberta_base_chinese/ not found!")

    # 读取数据
    data_path = "tcm_data_clean.json"
    with open(data_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    # 分词器
    tokenizer_bert = BertTokenizerFast.from_pretrained("bert_base_chinese", local_files_only=True)
    dataset = TCMEntityDataset(data_list, tokenizer_bert, max_len=128)

    # 划分训练/测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    # 1) EnhancedTCMER
    enhanced_tcmer_model = EnhancedTCMER(
        bert_name="bert_base_chinese",
        num_labels=len(LABEL2ID),
        lstm_hidden=768,
        num_lstm_layers=2,
        cnn_channels=256,
        dropout_prob=0.3
    )
    trainer_enhanced_tcmer = train_and_evaluate(enhanced_tcmer_model, train_ds, test_ds, "EnhancedTCMER")

    # 2) BertCRF
    class BertCRFForTokenClassification(nn.Module):
        def __init__(self, bert_name="bert_base_chinese", num_labels=5):
            super().__init__()
            self.num_labels = num_labels
            self.bertfc = BertForTokenClassification.from_pretrained(
                bert_name,
                num_labels=num_labels,
                local_files_only=True
            )
            self.crf = CRF(num_labels, batch_first=True)

        def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
            outputs = self.bertfc.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            seq_out = outputs.last_hidden_state
            logits = self.bertfc.dropout(seq_out)
            logits = self.bertfc.classifier(logits)

            loss = None
            if labels is not None:
                mask = attention_mask.bool()
                log_likelihood = self.crf(logits, labels, mask=mask, reduction="mean")
                loss = -log_likelihood
            return {"loss": loss, "logits": logits}

    transformer_crf_model = BertCRFForTokenClassification(
        bert_name="bert_base_chinese",
        num_labels=len(LABEL2ID)
    )
    trainer_trans_crf = train_and_evaluate(transformer_crf_model, train_ds, test_ds, "TransformerCRF")

    # 3) RoBERTaCRF
    class RobertaCRFForTokenClassification(nn.Module):
        def __init__(self, roberta_name="roberta_base_chinese", num_labels=5):
            super().__init__()
            self.num_labels = num_labels
            self.robertafc = RobertaForTokenClassification.from_pretrained(
                roberta_name,
                num_labels=num_labels,
                local_files_only=True
            )
            self.crf = CRF(num_labels, batch_first=True)

        def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
            outputs = self.robertafc.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            seq_out = outputs.last_hidden_state
            logits = self.robertafc.dropout(seq_out)
            logits = self.robertafc.classifier(logits)

            loss = None
            if labels is not None:
                mask = attention_mask.bool()
                log_likelihood = self.crf(logits, labels, mask=mask, reduction="mean")
                loss = -log_likelihood
            return {"loss": loss, "logits": logits}

    roberta_crf_model = RobertaCRFForTokenClassification(
        roberta_name="roberta_base_chinese",
        num_labels=len(LABEL2ID)
    )
    trainer_roberta_crf = train_and_evaluate(roberta_crf_model, train_ds, test_ds, "RoBERTaCRF")

    print("All training & evaluation done.")
