.PHONY: build test run clean

# Build the coordinator binary
build:
	go build -o coordinator

# Run tests
test:
	go test -v ./...

# Run the coordinator
run: build
	./coordinator

# Clean build artifacts
clean:
	rm -f coordinator
	go clean

# Format code
fmt:
	go fmt ./...

# Install dependencies
deps:
	go mod tidy
	go mod download
