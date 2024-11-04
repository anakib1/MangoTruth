package pkg

import (
	"errors"
	"github.com/spf13/viper"
	"log/slog"
	"os"
)

type Config struct {
	Compute ComputeConfig
	Engine  EngineConfig
	Logger  LoggerConfig
	Server  ServerConfig
}

type ServerConfig struct {
	Port string
}

type ComputeConfig struct {
	Host              string
	Port              string
	Username          string
	Password          string
	IdlePeriodSeconds int
	RequestQueueName  string
	ResponseQueueName string
}

type EngineConfig struct {
	FeedBufferSize    int
	ComputeBufferSize int
	IdlePeriodSeconds int
}

type LoggerConfig struct {
	Level  string
	Format string
}

func DefaultConfig() Config {
	return Config{
		Engine: EngineConfig{
			FeedBufferSize:    100,
			ComputeBufferSize: 100,
			IdlePeriodSeconds: 5,
		},
		Compute: ComputeConfig{
			Host:              "localhost",
			Port:              "5672",
			Username:          "guest",
			Password:          "guest",
			IdlePeriodSeconds: 5,
			RequestQueueName:  "request_queue",
			ResponseQueueName: "response_queue",
		},
		Logger: LoggerConfig{
			Level:  "INFO",
			Format: "json",
		},
		Server: ServerConfig{Port: "8080"},
	}
}

func MustGetConfig() *Config {
	cfgPath := getConfigPath(os.Getenv("APP_ENV"))
	defCfg := DefaultConfig()
	v, err := LoadConfig(cfgPath, "yml")
	if err != nil {
		slog.Error("Error in load defCfg",
			"err", err)
		return &defCfg
	}

	cfg, err := ParseConfig(v)
	if err != nil {
		slog.Error("Error in parse defCfg",
			"err", err)
		return &defCfg
	}

	return cfg
}

func getConfigPath(env string) string {
	if env == "docker" {
		return "/core/config/config-docker"
	} else if env == "production" {
		return "/core/config/config-production"
	} else {
		return "/core/config/config-development"
	}
}

func ParseConfig(v *viper.Viper) (*Config, error) {
	cfg := DefaultConfig()
	err := v.Unmarshal(&cfg)
	if err != nil {
		slog.Error("Unable to parse config",
			"err", err)
		return nil, err
	}
	return &cfg, nil
}

func LoadConfig(filename string, fileType string) (*viper.Viper, error) {
	v := viper.New()
	v.SetConfigType(fileType)
	v.SetConfigName(filename)
	v.AddConfigPath(".")
	v.AutomaticEnv()

	err := v.ReadInConfig()
	if err != nil {
		slog.Error("Unable to read config",
			"err", err)
		var configFileNotFoundError viper.ConfigFileNotFoundError
		if errors.As(err, &configFileNotFoundError) {
			return nil, errors.New("config file not found")
		}
		return nil, err
	}
	return v, nil
}