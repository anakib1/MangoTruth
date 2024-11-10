FROM golang:bookworm

WORKDIR /mango
COPY . .

RUN ls

RUN go mod tidy
EXPOSE 8080
CMD ["go", "run", "./cmd/server/main.go"]
