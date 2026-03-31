PYTHON=python

validate:
	$(PYTHON) -m marketing_intelligence.cli validate --data-dir data/raw

features:
	$(PYTHON) -m marketing_intelligence.cli build-features --project-root .

train:
	$(PYTHON) -m marketing_intelligence.cli train --project-root .

analysis:
	$(PYTHON) -m marketing_intelligence.cli analyze --project-root .

pipeline: validate features train analysis

test:
	pytest -q
