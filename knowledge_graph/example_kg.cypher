MATCH (h:HerbalMedicine)-[r:TREATS]->(s:Symptom {name: 'Anemia'})
RETURN h, r, s;
