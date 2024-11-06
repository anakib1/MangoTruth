FROM golang:bookworm

WORKDIR /mango
COPY . .

RUN ls

RUN go mod tidy

CMD ["go", "run", "./cmd/server/main.go"]
