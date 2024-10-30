package main

import (
	"log"
	"log/slog"
	_ "mango_truth/core/docs"
	engine2 "mango_truth/core/pkg/engine"
	"mango_truth/core/pkg/web"
	"os"
)

func main() {

	restToEngine := make(chan any)

	slog.SetLogLoggerLevel(slog.Level(-4))

	opts := &slog.HandlerOptions{Level: slog.LevelDebug}
	logger := slog.New(slog.NewJSONHandler(os.Stdout, opts))
	slog.SetDefault(logger)

	go engine2.NewMangoEngine(restToEngine).Work()
	log.Fatal(web.NewMangoRest(restToEngine).Run(":8080"))

}
