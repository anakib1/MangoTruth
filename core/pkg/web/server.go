package web

import (
	"github.com/gin-gonic/gin"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
	"mango_truth/pkg/modules"
	"mango_truth/pkg/storage"
)

type MangoRest struct {
	*gin.Engine
	storage    *storage.Storage
	engineSink chan<- modules.ClientToServer
}

func NewMangoRest(engineSink chan<- modules.ClientToServer, storageRef *storage.Storage) *MangoRest {
	r := &MangoRest{
		Engine:     gin.Default(), // Initialize the Gin engineSink
		engineSink: engineSink,
		storage:    storageRef,
	}

	v1 := r.Group("/api/v1")
	{
		v1.GET("/detection", r.GetDetection)
		v1.PUT("/detection", r.PutDetection)
		v1.GET("/detectors", r.GetDetectors)
	}

	r.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))
	return r
}
