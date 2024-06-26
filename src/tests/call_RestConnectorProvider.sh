curl -i -X GET --http1.1 "localhost:9698/hitec/classify/relevance/status" \
-H "Content-Type: application/json"

curl -i -X POST --http1.1 "localhost:9698/hitec/classify/relevance/run" \
-H "Content-Type: application/json" \
--data-binary "@src/tests/testContent.json"
