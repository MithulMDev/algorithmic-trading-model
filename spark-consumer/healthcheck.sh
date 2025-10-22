#!/bin/bash
# Health check script for OHLCV services
# Usage: healthcheck.sh [spark-consumer|signal-processor|portfolio-manager]

SERVICE_TYPE=${1:-spark-consumer}

check_process() {
    local process_name=$1
    pgrep -f "$process_name" > /dev/null 2>&1
    return $?
}

check_redis_connection() {
    python3 -c "
import redis
import os
try:
    r = redis.Redis(
        host=os.getenv('REDIS_HOST', 'redis'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=int(os.getenv('REDIS_DB', 0)),
        socket_timeout=2
    )
    r.ping()
    exit(0)
except Exception as e:
    exit(1)
" 2>/dev/null
    return $?
}

case $SERVICE_TYPE in
    spark-consumer)
        # Check if Spark process is running
        if check_process "spark_consumer.py"; then
            # Check if it can connect to Redis (where it writes data)
            if check_redis_connection; then
                exit 0
            else
                echo "Spark consumer running but Redis connection failed"
                exit 1
            fi
        else
            echo "Spark consumer process not found"
            exit 1
        fi
        ;;
    
    signal-processor)
        # Check if signal processor is running
        if check_process "signal_processor.py"; then
            # Check if it can connect to Redis (where it reads data)
            if check_redis_connection; then
                exit 0
            else
                echo "Signal processor running but Redis connection failed"
                exit 1
            fi
        else
            echo "Signal processor process not found"
            exit 1
        fi
        ;;
    
    portfolio-manager)
        # Check if portfolio manager is running
        if check_process "portfolio_manager.py"; then
            # Check if it can connect to Redis
            if check_redis_connection; then
                exit 0
            else
                echo "Portfolio manager running but Redis connection failed"
                exit 1
            fi
        else
            echo "Portfolio manager process not found"
            exit 1
        fi
        ;;
    
    *)
        echo "Unknown service type: $SERVICE_TYPE"
        exit 1
        ;;
esac