pipeline {
    agent any

    environment {
        MPLCONFIGDIR = '/tmp/matplotlib'
        MLFLOW_DISABLE_AGENT_HINT = '1'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Setup uv & Dependencies') {
            steps {
                sh '''
                    # Ensure uv is installed on Jenkins agent
                    if ! command -v uv >/dev/null 2>&1; then
                        echo "Installing uv..."
                        curl -LsSf https://astral.sh/uv/install.sh | sh
                        export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
                    fi
                    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
                    uv --version
                    uv sync
                '''
            }
        }

        stage('Data Schema & Integrity Tests') {
            steps {
                sh '''
                    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
                    uv run pytest tests/test_data.py -v --junitxml=reports/junit-data-tests.xml
                '''
            }
        }

        stage('Model Training & MLflow Governance') {
            steps {
                sh '''
                    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
                    uv run python src/models/train.py --gate 0.75
                '''
            }
        }

        stage('Model Quality & Latency Gates') {
            steps {
                sh '''
                    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
                    uv run pytest tests/test_model.py -v --junitxml=reports/junit-model-tests.xml
                '''
            }
        }

        stage('Evidently Drift Monitoring') {
            steps {
                sh '''
                    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
                    uv run python src/monitoring/simulate_drift.py
                    uv run python src/monitoring/drift_report.py
                '''
            }
        }
    }

    post {
        always {
            junit allowEmptyResults: true, testResults: 'reports/junit-*.xml'
            archiveArtifacts artifacts: 'reports/**', allowEmptyArchive: true
        }
        success {
            echo 'FinTech Credit Risk MLOps Pipeline Succeeded! All gates passed.'
        }
        failure {
            echo 'Pipeline failed. Check stage logs for failed assertions or errors.'
        }
    }
}
