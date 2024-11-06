package main

import (
	"log"
	"log/slog"
	_ "mango_truth/docs"
	"mango_truth/pkg"
	"mango_truth/pkg/core"
	"mango_truth/pkg/utils"
	"mango_truth/pkg/web"
)

func main() {

	cfg := pkg.MustGetConfig()
	utils.ConfigureLogging(&cfg.Logger)
	slog.Info("Running with config:", "config", cfg)

	engine, requests, statuses, restToEngine := core.NewMangoEngine(cfg.Engine)
	go engine.Work()
	router := core.NewComputeRouter(cfg.Compute, requests, statuses)
	go router.Work()

	log.Fatal(web.NewMangoRest(restToEngine).Run(":" + cfg.Server.Port))

}
