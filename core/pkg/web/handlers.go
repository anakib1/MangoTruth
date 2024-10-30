package web

import (
	"github.com/gin-gonic/gin"
	"mango_truth/core/pkg/modules"
	"net/http"
	"time"
)

// GetDetection handles GET /api/v1/detection
// @Summary Get the detection status
// @Description Fetch the detection status and verdict by requestId.
// @Produce  json
// @Param requestId query string true "Request ID" format(uuid)
// @Success 200 {object} modules.DetectionStatus
// @Failure 400 {object} map[string]string "Missing requestId parameter"
// @Failure 404 {object} map[string]string "Request not found"
// @Router /detection [get]
func (r *MangoRest) GetDetection(c *gin.Context) {
	var req modules.DetectionQuery
	if err := c.ShouldBindBodyWithJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Wrong format of DetectionQuery"})
		return
	}

	r.waitFromEngine(c, modules.DetectionQueryEx{
		DetectionQuery: req,
		ClientToServer: &modules.ClientToServer{},
	})
}

// PutDetection handles PUT /api/v1/detection
// @Summary Submit a detection request
// @Description Submit a detection request for processing.
// @Accept  json
// @Produce  json
// @Param detectionRequest body modules.DetectionRequest true "Detection Request"
// @Success 200 {object} modules.DetectionStatus
// @Failure 400 {object} map[string]string "Invalid request body"
// @Router /detection [put]
func (r *MangoRest) PutDetection(c *gin.Context) {
	var req modules.DetectionRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request body"})
		return
	}

	r.waitFromEngine(c, modules.DetectionRequestEx{
		DetectionRequest: req,
		ClientToServer:   &modules.ClientToServer{},
	})

}

func (r *MangoRest) waitFromEngine(c *gin.Context, req modules.Callbacker) {
	resp := make(chan modules.DetectionStatus)
	req.DefineCallback(resp)

	r.engine <- req

	select {
	case res := <-resp:
		c.JSON(http.StatusOK, res)
	case <-time.After(time.Second * 5):
		c.JSON(http.StatusRequestTimeout, gin.H{"error": "Request timeout"})
	}
}
