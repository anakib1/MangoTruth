package core

import (
	"github.com/google/uuid"
	"mango_truth/pkg/modules"
	"sync"
)

type Storage struct {
	cache sync.Map
}

func (s *Storage) updStatus(status modules.DetectionStatus) {
	s.cache.Store(status.RequestId, status)
}

func (s *Storage) getStatus(id uuid.UUID) modules.DetectionStatus {

	value, ok := s.cache.Load(id)
	if !ok {
		return modules.DetectionStatus{
			RequestId: id,
			Status:    "UNKNOWN",
		}
	}
	return value.(modules.DetectionStatus)
}
