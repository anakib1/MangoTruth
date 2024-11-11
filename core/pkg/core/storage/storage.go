package storage

import (
	"context"
	"fmt"
	"encoding/json"
	"database/sql"
	"mango_truth/pkg/core/storage/models"
	"mango_truth/pkg/modules"
	"github.com/google/uuid"
	_ "github.com/lib/pq"
	"github.com/volatiletech/null/v8"
	"github.com/volatiletech/sqlboiler/v4/boil"
)

type Storage struct {
	db boil.ContextExecutor
}

func NewStorage() *Storage {
	db, err := sql.Open("postgres", "dbname=mango-db host=postgres user=mango-user password=password sslmode=disable")
	if err != nil {
		panic(fmt.Sprintf("Can not connect to the database: %s", err.Error()))
	}
	return &Storage{db: db}
}

func (s *Storage) UpdStatus(status modules.DetectionStatus) {
	data, err := json.Marshal(status.Verdict)
	if err != nil {
		panic(fmt.Sprintf("Can not connect verdict to json in DetectionStatus with requestId=%s, error: %s", status.RequestId, err.Error()))
	}
	new_status := models.DetectionStatus{
		RequestID: status.RequestId.String(),
		Status: status.Status,
		Data: null.BytesFrom(data),
	}

	err = new_status.Insert(context.TODO(), s.db, boil.Infer())
	if err != nil {
		panic(fmt.Sprintf("Can not insert value into storage, error: %s", err.Error()))
	}
}

func (s *Storage) GetStatus(id uuid.UUID) modules.DetectionStatus {
	status, err := models.FindDetectionStatus(context.TODO(), s.db, id.String())
	switch err {
	case nil:
		// Do nothing
	case sql.ErrNoRows:
		return modules.DetectionStatus{RequestId: id, Status: "UNKNOWN",}
	default:
		panic(fmt.Sprintf("Error in FindDetectionStatus, error: %s", err.Error()))
	}
	request_id, err := uuid.Parse(status.RequestID)
	if err != nil {
		panic(fmt.Sprintf("RequestID is in the wrong format, err: %s", err.Error()))
	}
	var verdict modules.Verdict
	err = json.Unmarshal(status.Data.Bytes, &verdict)
	if err != nil {
		panic(fmt.Sprintf("Error parsing []bytes to Verdict, error: %s", err.Error()))
	}
	return modules.DetectionStatus{RequestId: request_id, Status: status.Status, Verdict: verdict,}
}
