from flask import Flask, Response, json, jsonify, request
from flask_cors import CORS

from main.behavior.RelevanceClassifier import RelevanceClassifier
from main.tooling.Logger import logging_setup

logger = logging_setup(__name__)

app = Flask(__name__)
#  cors only for local testing
cors = CORS(app)


class ComponentRelevanceClassifierRestConnectorProvider():
    """
        Description: Entrypoint for calling the RelevanceClassificationService.
    """

    @app.route("/hitec/classify/relevance/run", methods=["POST"])
    def classify_relevance() -> Response:  # type: ignore
        app.logger.debug("/hitec/classify/relevance/run called")

        content = json.loads(request.data.decode("utf-8"))

        relevanceClassifier = RelevanceClassifier()

        resultMessage = relevanceClassifier.startCreationPipeline(content)

        result = dict()
        result.update({"message": resultMessage})
        app.logger.info(result)

        return jsonify(result)

    @app.route("/hitec/classify/relevance/status", methods=["GET"])
    def get_status() -> Response:  # type: ignore
        try:
            app.logger.info("Status requested")
            status = {
                "status": "operational",
            }
        except Exception as e:
            status = {"status": "not_operational", "error": str(e)}

        return jsonify(status)

    if __name__ == "__main__":
        app.run(debug=True, host="0.0.0.0", port=9123)
