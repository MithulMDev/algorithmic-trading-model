# Trading System Architecture Diagrams

This package contains comprehensive Mermaid diagrams documenting the architecture, data flow, and communication patterns of your trading system.

## System Overview

Your system consists of:
- **Kafka Producer** → Reads from MongoDB and streams OHLCV data
- **spark_consumer.py** → Main pipeline coordinator
- **indicator_engine.py** → Calculates 14 technical indicators with 60-row warmup
- **signal_processor.py** → Generates 6 trading signals (runs every 2 seconds)
- **portfolio_manager.py** → Manages portfolio and generates trade orders (runs every 5 minutes)

## Diagram Descriptions

### 01_system_architecture.mmd
**High-level system architecture showing all components and their connections**
- Shows the complete data flow from MongoDB → Kafka → Consumer → Indicator Engine → Storage
- Highlights parallel processors (signal_processor and portfolio_manager)
- Shows all storage layers: Redis (DB 0 & 1), InfluxDB, ClickHouse

### 02_redis_data_structure.mmd
**Redis key structure and access patterns**
- Documents Redis DB 0 (latest OHLCV) and DB 1 (indicators, signals, portfolio)
- Shows which scripts read/write which keys
- Demonstrates the data separation strategy

### 03_communication_sequence.mmd
**Detailed sequence diagram of component interactions**
- Shows the message flow between all components
- Includes warm-up phase logic
- Demonstrates parallel processing loops
- Shows timing of operations

### 04_data_transformation.mmd
**Data enrichment pipeline from raw to processed**
- Tracks data transformation through 7 stages
- Shows what fields are added at each stage
- Visualizes the warm-up to enriched transition
- Demonstrates signal and order generation

### 05_parallel_processing.mmd
**Parallel execution architecture**
- Shows main stream processing (continuous)
- Shows signal processor loop (every 2 seconds)
- Shows portfolio manager loop (every 5 minutes)
- Illustrates how they share Redis but run independently

### 06_indicator_engine_workflow.mmd
**Indicator engine internal logic**
- Details the 60-row warm-up mechanism
- Shows the decision flow for returning data
- Explains first-ready vs already-warmed behavior
- Documents all 14 calculated indicators

### 07_signal_processor_workflow.mmd
**Signal generation process**
- Shows parallel processing with 2 workers
- Details the 6 signal calculations
- Demonstrates regime classification logic
- Shows enrichment validation checks

### 08_portfolio_manager_workflow.mmd
**Portfolio management cycle**
- Shows signal loading and classification
- Details target weight calculation
- Demonstrates order generation logic
- Shows validation and prioritization steps

### 09_redis_detailed_schema.mmd
**Complete Redis data model with all fields**
- Lists every field in every Redis key
- Shows data types and structures
- Documents JSON-nested structures
- Maps operations to keys

### 10_timing_execution.mmd
**Gantt chart showing execution timeline**
- Visualizes concurrent processing
- Shows timing of each component
- Demonstrates Redis access patterns over time
- Illustrates the 2s and 5min intervals

### 11_data_lifecycle.mmd
**State machine of data flow through the system**
- Shows complete lifecycle from MongoDB to orders
- Includes decision points and conditional flows
- Documents warm-up states
- Shows parallel processing branches

### 12_component_dependencies.mmd
**Dependency graph of all components**
- Shows what each component depends on
- Highlights independent vs dependent processors
- Documents shared storage dependencies
- Maps the execution order requirements

## How to Use These Diagrams

### Viewing Mermaid Diagrams

1. **Online Viewers:**
   - [Mermaid Live Editor](https://mermaid.live)
   - [GitHub](https://github.com) - GitHub renders .mmd files automatically

2. **VS Code:**
   - Install "Markdown Preview Mermaid Support" extension
   - Open any .mmd file and preview

3. **Command Line:**
   ```bash
   # Install mermaid-cli
   npm install -g @mermaid-js/mermaid-cli
   
   # Convert to PNG/SVG
   mmdc -i 01_system_architecture.mmd -o diagram.png
   ```

## Key Redis Keys Reference

### Redis DB 0
- `ohlcv:latest:{SYMBOL}` - Latest OHLCV tick for each symbol

### Redis DB 1
- `indicator:history:{SYMBOL}` - 60-row sliding window with indicators
- `signals:latest:{SYMBOL}` - Current trading signals
- `portfolio:state` - Portfolio state (cash, positions, history)
- `trade:orders:{TIMESTAMP}` - Generated trade orders

## Data Flow Summary

```
MongoDB → Kafka → spark_consumer → indicator_engine (warm-up 60 rows)
                                  ↓
                    [Redis DB 1: indicator:history:*]
                                  ↓
                    [Redis DB 0: ohlcv:latest:*]
                    [InfluxDB: time-series]
                    [ClickHouse: analytics]
                                  ↓
                    signal_processor (every 2s) → [Redis DB 1: signals:latest:*]
                                  ↓
                    portfolio_manager (every 5 min) → [Redis DB 1: portfolio:state]
                                                    → [Redis DB 1: trade:orders:*]
```

## System Characteristics

- **Warm-up Phase**: 60 rows needed before indicators are calculated
- **Signal Generation**: 6 independent signals combined into final score
- **Parallel Processing**: Signal processor and portfolio manager run independently
- **Data Storage**: Triple redundancy (Redis, InfluxDB, ClickHouse)
- **State Management**: All state in Redis for cross-component communication

## For More Information

Each diagram file contains additional inline documentation. Open them in a Mermaid viewer to see the full details with proper formatting and colors.
