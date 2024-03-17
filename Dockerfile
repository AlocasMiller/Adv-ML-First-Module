FROM python:3.10
WORKDIR /adv-ml-first-module
COPY . /adv-ml-first-module
RUN pip install ./dist/adv_ml_first_module-0.1.0-py3-none-any.whl
CMD ["python", "adv_ml_first_module/main.py"]