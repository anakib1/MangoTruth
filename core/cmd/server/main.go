package main

import (
	"log"
	"log/slog"
	_ "mango_truth/core/docs"
	"mango_truth/core/pkg"
	engine2 "mango_truth/core/pkg/engine"
	"mango_truth/core/pkg/utils"
	"mango_truth/core/pkg/web"
)

func main() {

	cfg := pkg.MustGetConfig()
	utils.ConfigureLogging(&cfg.Logger)
	slog.Info("Running with config:", "config", cfg)

	restToEngine := make(chan any, cfg.Engine.FeedBufferSize)

	engine, requests, statuses := engine2.NewMangoEngine(cfg.Engine, restToEngine)
	go engine.Work()
	router := engine2.NewComputeRouter(cfg.Compute, requests, statuses)
	go router.Work()

	log.Fatal(web.NewMangoRest(restToEngine).Run(":" + cfg.Server.Port))

}
