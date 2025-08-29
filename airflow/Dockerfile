FROM apache/airflow:2.8.1-python3.10
ENV AIRFLOW_HOME=/opt/airflow \
    INFERENCE_STORE_DIR=/opt/airflow/inference_store
USER root
RUN mkdir -p ${AIRFLOW_HOME}/dags ${AIRFLOW_HOME}/logs ${AIRFLOW_HOME}/plugins ${INFERENCE_STORE_DIR} \
    && chown -R airflow: ${AIRFLOW_HOME}
USER airflow
COPY requirements.txt ${AIRFLOW_HOME}/requirements.txt
RUN pip install --no-cache-dir -r ${AIRFLOW_HOME}/requirements.txt