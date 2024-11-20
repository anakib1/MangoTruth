package web

import (
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"mango_truth/pkg/modules"
	"net/http"
	"time"
)

// GetDetection handles GET /api/v1/detection
// @Summary Get the detection status
// @Description Fetch the detection status and verdict by requestId.
// @Produce  json
// @Accept  json
// @Param detectionQuery body modules.DetectionQuery true "Detection Query Request"
// @Success 200 {object} modules.DetectionStatus
// @Failure 400 {object} map[string]string "Missing requestId parameter"
// @Failure 404 {object} map[string]string "Request not found"
// @Router /api/v1/detection [get]
func (r *MangoRest) GetDetection(c *gin.Context) {
	var req modules.DetectionQuery
	if err := c.ShouldBindBodyWithJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Wrong format of DetectionQuery"})
		return
	}

	r.waitFromEngine(c, modules.ClientToServer{Msg: req})
}

// PutDetection handles PUT /api/v1/detection
// @Summary Submit a detection request
// @Description Submit a detection request for processing.
// @Accept  json
// @Produce  json
// @Param detectionRequest body modules.DetectionRequest true "Detection Request"
// @Success 200 {object} modules.DetectionStatus
// @Failure 400 {object} map[string]string "Invalid request body"
// @Router /api/v1/detection [put]
func (r *MangoRest) PutDetection(c *gin.Context) {
	var req modules.DetectionRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request body"})
		return
	}

	if req.RequestId != uuid.Nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "RequestId will be specified by server"})
		return
	}

	if !r.storage.DetectorExists(req.DetectorName) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Detected not exists"})
		return
	}

	req.RequestId = uuid.New()

	r.waitFromEngine(c, modules.ClientToServer{Msg: req})

}

func (r *MangoRest) GetDetectors(c *gin.Context) {
	detectors := r.storage.GetDetectors()
	c.JSON(http.StatusOK, gin.H{"detectors": detectors})
}

func (r *MangoRest) waitFromEngine(c *gin.Context, req modules.ClientToServer) {
	resp := make(chan modules.DetectionStatus)
	req.Ret = resp

	r.engineSink <- req

	select {
	case res := <-resp:
		c.JSON(http.StatusOK, res)
	case <-time.After(time.Second * 5):
		c.JSON(http.StatusRequestTimeout, gin.H{"error": "Request timeout"})
	}
}
