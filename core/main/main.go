package main

import (
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"log"
	"net/http"
	"sync"

	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
	_ "mango_truth/core/docs" // Adjust the import path to match the generated docs location
)

// DetectionRequest represents a detection request
// @Description DetectionRequest contains the UUID and content to be analyzed
type DetectionRequest struct {
	// RequestID is the unique identifier for the detection request
	// required: true
	// example: f47ac10b-58cc-4372-a567-0e02b2c3d479
	RequestID uuid.UUID `json:"requestId" example:"f47ac10b-58cc-4372-a567-0e02b2c3d479"`

	// Content to be analyzed for detection
	// required: true
	// example: This is the content to be analyzed.
	Content string `json:"content" example:"This is the content to be analyzed."`
}

// DetectionStatus represents the status and result of the detection
// @Description DetectionStatus includes the status of the request and the verdict
type DetectionStatus struct {
	// Status is the current status of the detection request
	// Enum: "REJECTED" "FAILED" "IN_PROGRESS" "FINISHED"
	// required: true
	// example: IN_PROGRESS
	Status string `json:"status" example:"IN_PROGRESS"`

	// Verdict is the result of the detection (optional)
	// It can be null if no verdict is available yet
	Verdict *Verdict `json:"verdict"`
}

// Verdict represents the result of the detection
type Verdict struct {
	// Labels with associated probabilities
	Labels []Label `json:"labels"`
}

// Label represents a classification label with its probability
type Label struct {
	// Label name (e.g., "plagiarism" or "ai-generated")
	// example: plagiarism
	Label string `json:"label" example:"plagiarism"`

	// Probability of the label being true
	// example: 0.85
	Probability float64 `json:"probability" example:"0.85"`
}

// In-memory storage for detection requests and statuses
var detectionStore = make(map[string]*DetectionStatus)
var mu sync.Mutex

// @title Detection API
// @version 1.0
// @description This is a detection system API to process requests and return detection status.
// @host localhost:8080
// @BasePath /api/v1
func main() {
	r := gin.Default()

	// Swagger endpoint
	r.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))

	r.PUT("/api/v1/detection", putDetection)
	r.GET("/api/v1/detection", getDetection)

	log.Println("Starting server on :8080...")
	r.Run(":8080")
}

// handlePutDetection handles PUT /api/v1/detection
// @Summary Submit a detection request
// @Description Submit a detection request for processing.
// @Accept  json
// @Produce  json
// @Param detectionRequest body DetectionRequest true "Detection Request"
// @Success 200 {object} DetectionStatus
// @Failure 400 {object} map[string]string "Invalid request body"
// @Router /detection [put]
func putDetection(c *gin.Context) {
	var req DetectionRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request body"})
		return
	}

	mu.Lock()
	defer mu.Unlock()

	// Check if requestId is valid, generate if missing
	if req.RequestID == uuid.Nil {
		req.RequestID = uuid.New()
	}

	// Simulate detection status
	status := "IN_PROGRESS"
	if len(req.Content) == 0 {
		status = "REJECTED"
	}

	// Store in the detection store
	detectionStore[req.RequestID.String()] = &DetectionStatus{
		Status: status,
		Verdict: &Verdict{
			Labels: []Label{
				{"AI Generated", 0.85},
				{"Human generated", 0.15},
			},
		},
	}

	// Respond with status and verdict
	c.JSON(http.StatusOK, detectionStore[req.RequestID.String()])
}

// handleGetDetection handles GET /api/v1/detection
// @Summary Get the detection status
// @Description Fetch the detection status and verdict by requestId.
// @Produce  json
// @Param requestId query string true "Request ID" format(uuid)
// @Success 200 {object} DetectionStatus
// @Failure 400 {object} map[string]string "Missing requestId parameter"
// @Failure 404 {object} map[string]string "Request not found"
// @Router /detection [get]
func getDetection(c *gin.Context) {
	requestID := c.Query("requestId")
	if requestID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Missing requestId parameter"})
		return
	}

	mu.Lock()
	defer mu.Unlock()

	status, exists := detectionStore[requestID]
	if !exists {
		c.JSON(http.StatusNotFound, gin.H{"error": "Request not found"})
		return
	}

	c.JSON(http.StatusOK, status)
}
