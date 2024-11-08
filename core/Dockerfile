FROM golang:bookworm

WORKDIR /mango
COPY ./core ./core
COPY go.* .

RUN go mod tidy
EXPOSE 8080
CMD ["go", "run", "./core/cmd/server/main.go"]
