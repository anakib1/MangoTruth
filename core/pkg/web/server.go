package web

import (
	"github.com/gin-gonic/gin"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
)

type MangoRest struct {
	*gin.Engine
	engine chan<- any
}

func NewMangoRest(engine chan<- any) *MangoRest {
	r := &MangoRest{
		Engine: gin.Default(), // Initialize the Gin engine
		engine: engine,
	}

	v1 := r.Group("/api/v1")
	{
		v1.GET("/detection", r.GetDetection)
		v1.PUT("/detection", r.PutDetection)
	}

	r.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))
	return r
}
