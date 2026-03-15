VERSION := $(shell python -c "import gitdb; print(gitdb.__version__)")
DOCKER_REPO := vincentkaufmann/gitdb

.PHONY: build test binary-macos binary-linux docker docker-push release clean

# ── Development ───────────────────────────────────────────
build:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -x -q

# ── Binaries ──────────────────────────────────────────────
binary-macos:
	pyinstaller gitdb.spec --clean --noconfirm
	cp dist/gitdb dist/gitdb-macos-arm64
	@echo "Built: dist/gitdb-macos-arm64 ($$(du -h dist/gitdb-macos-arm64 | cut -f1))"

binary-linux:
	docker build -f Dockerfile.build-linux -t gitdb-build-linux .
	docker create --name gitdb-extract gitdb-build-linux
	docker cp gitdb-extract:/app/dist/gitdb dist/gitdb-linux-amd64
	docker rm gitdb-extract
	@echo "Built: dist/gitdb-linux-amd64 ($$(du -h dist/gitdb-linux-amd64 | cut -f1))"

# ── Docker ────────────────────────────────────────────────
docker:
	docker build -t $(DOCKER_REPO):$(VERSION) -t $(DOCKER_REPO):latest --target base .
	@echo "Built: $(DOCKER_REPO):$(VERSION)"

docker-full:
	docker build -t $(DOCKER_REPO):$(VERSION)-full -t $(DOCKER_REPO):full --target full .
	@echo "Built: $(DOCKER_REPO):$(VERSION)-full"

docker-push: docker docker-full
	docker push $(DOCKER_REPO):$(VERSION)
	docker push $(DOCKER_REPO):latest
	docker push $(DOCKER_REPO):$(VERSION)-full
	docker push $(DOCKER_REPO):full

# ── PyPI ──────────────────────────────────────────────────
pypi:
	rm -rf dist/*.whl dist/*.tar.gz
	python -m build
	python -m twine upload dist/gitdb_vectors-$(VERSION)*

# ── GitHub Release ────────────────────────────────────────
release:
	gh release create v$(VERSION) \
		--title "GitDB v$(VERSION)" \
		--generate-notes \
		dist/gitdb-macos-arm64 \
		dist/gitdb-linux-amd64 \
		|| echo "Build binaries first: make binary-macos binary-linux"

# ── Clean ─────────────────────────────────────────────────
clean:
	rm -rf dist/ build/ *.egg-info __pycache__
