package modules

import (
	"github.com/google/uuid"
)

type DetectionQuery struct {
	// RequestID is the unique identifier for the detection request
	// required: true
	// example: f47ac10b-58cc-4372-a567-0e02b2c3d479
	RequestId uuid.UUID `json:"requestId" binding:"required" example:"f47ac10b-58cc-4372-a567-0e02b2c3d479"`
}

// DetectionRequest represents a detection request
// @Description DetectionRequest contains the request UUID and content to be analyzed
type DetectionRequest struct {
	// Is generated
	RequestId uuid.UUID `json:"-"`

	// Content to be analyzed for detection
	// required: true
	// example: This is the content to be analyzed.
	Content string `json:"content" example:"This is the content to be analyzed."`
}

// DetectionStatus represents the status and result of the detection
// @Description DetectionStatus includes the status of the request and the verdict
type DetectionStatus struct {
	// RequestID is the unique identifier for the detection request
	// required: true
	// example: f47ac10b-58cc-4372-a567-0e02b2c3d479
	RequestId uuid.UUID `json:"requestId" binding:"required" example:"f47ac10b-58cc-4372-a567-0e02b2c3d479"`

	// Status is the current status of the detection request
	// Enum: "REJECTED" "FAILED" "IN_PROGRESS" "FINISHED"
	// required: true
	// example: IN_PROGRESS
	Status string `json:"status" example:"IN_PROGRESS"`

	// Verdict is the result of the detection (optional)
	// It can be null if no verdict is available yet
	Verdict Verdict `json:"verdict"`
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

type ClientToServer struct {
	Ret chan<- DetectionStatus
	Msg any
}