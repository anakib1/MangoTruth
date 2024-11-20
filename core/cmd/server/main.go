package main

import (
	"log"
	"log/slog"
	_ "mango_truth/docs"
	"mango_truth/pkg"
	"mango_truth/pkg/core"
	"mango_truth/pkg/storage"
	"mango_truth/pkg/utils"
	"mango_truth/pkg/web"
)

func main() {

	cfg := pkg.MustGetConfig()
	utils.ConfigureLogging(&cfg.Logger)
	slog.Info("Running with config:", "config", cfg)

	storageRef := storage.NewStorage(cfg.Storage)
	engine, requests, statuses, restToEngine := core.NewMangoEngine(cfg.Engine, storageRef)
	go engine.Work()
	router := core.NewComputeRouter(cfg.Compute, requests, statuses)
	go router.Work()

	log.Fatal(web.NewMangoRest(restToEngine, storageRef).Run(":" + cfg.Server.Port))

}
