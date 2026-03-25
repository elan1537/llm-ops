# Paper Pipeline Design

## 목적

PDF 논문을 업로드하면 OCR → 마크다운 변환 → 저장 → 열람할 수 있는 웹 앱.
기존 2-stage OCR 파이프라인(DeepSeek + Qwen) 위에 구축.

## 아키텍처

```
Browser (HTMX + Jinja2)
    ↓
FastAPI Backend (:8080)
    ├→ PDF 업로드 + 백그라운드 OCR
    │    ├→ DeepSeek-OCR (:8003) — 텍스트 + figure bbox
    │    └→ Qwen (:8000) — figure description
    ├→ SQLite DB (논문 메타데이터)
    ├→ 파일 저장 (마크다운 + 이미지)
    └→ Notion API (선택, 자동 전송)
```

## 핵심 페이지

### 1. 목록 페이지 (`/`)

- 업로드된 논문 리스트 (제목, 날짜, 상태, 페이지 수)
- 드래그앤드롭 PDF 업로드
- 텍스트 검색 (제목/내용)
- 상태 표시: processing → done → (옵션) synced to Notion

### 2. 뷰어 페이지 (`/papers/{id}`)

- 변환된 마크다운 렌더링
- figure 이미지 인라인 표시
- 원본 PDF 다운로드 링크
- 페이지별 네비게이션

### 3. 상태 페이지 (HTMX 실시간)

- OCR 진행률 (현재 페이지 / 전체 페이지)
- 에러 시 실패 메시지 표시
- 완료 시 뷰어로 자동 이동

## 데이터 모델

### papers 테이블 (SQLite)

```sql
CREATE TABLE papers (
    id          TEXT PRIMARY KEY,   -- UUID
    title       TEXT NOT NULL,      -- 파일명 또는 추출된 제목
    filename    TEXT NOT NULL,      -- 원본 PDF 파일명
    status      TEXT NOT NULL,      -- uploading, processing, done, error
    page_count  INTEGER DEFAULT 0,
    current_page INTEGER DEFAULT 0, -- OCR 진행 상황
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME,
    error_msg   TEXT,
    notion_page_id TEXT,            -- Notion 연동 시 페이지 ID
    file_size   INTEGER DEFAULT 0   -- PDF 파일 크기 (bytes)
);
```

### 파일 저장 구조

```
data/papers/{paper_id}/
    ├── original.pdf           # 원본 PDF
    ├── output.md              # 변환된 전체 마크다운
    ├── metadata.json          # 메타데이터 (페이지 수, figure 수 등)
    └── images/                # 추출된 figure 이미지
        ├── page0_fig0.jpg
        ├── page1_fig0.jpg
        └── ...
```

## 데이터 흐름

```
1. 사용자가 PDF 업로드 (드래그앤드롭 또는 파일 선택)
2. Backend:
   a. UUID 생성, DB에 status="uploading" 등록
   b. PDF를 data/papers/{id}/original.pdf에 저장
   c. status="processing" 업데이트
   d. 백그라운드 태스크로 OCR 시작
3. OCR Worker (백그라운드):
   a. PDF → 페이지별 이미지 (PyMuPDF)
   b. DeepSeek-OCR로 텍스트 + figure bbox 추출
   c. Figure crop → Qwen으로 description 생성
   d. 페이지별 진행 상황 DB 업데이트 (current_page)
   e. 전체 마크다운 output.md에 저장
   f. metadata.json 저장
   g. status="done" 업데이트
4. 프론트엔드:
   - HTMX로 /papers/{id}/status를 2초마다 폴링
   - 진행률 표시 (current_page / page_count)
   - 완료 시 뷰어 페이지로 이동
5. (옵션) Notion 연동:
   - 완료 후 자동으로 Notion 페이지 생성
   - 마크다운 내용을 Notion blocks으로 변환
```

## API 엔드포인트

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | 논문 목록 페이지 |
| `POST` | `/upload` | PDF 업로드 → OCR 시작 |
| `GET` | `/papers/{id}` | 논문 뷰어 페이지 |
| `GET` | `/papers/{id}/status` | OCR 진행 상황 (HTMX partial) |
| `GET` | `/papers/{id}/download` | 원본 PDF 다운로드 |
| `GET` | `/papers/{id}/markdown` | 변환된 마크다운 raw 텍스트 |
| `DELETE` | `/papers/{id}` | 논문 삭제 |
| `POST` | `/papers/{id}/sync-notion` | Notion에 수동 동기화 |
| `GET` | `/search?q=...` | 논문 검색 (HTMX partial) |

## 기술 스택

| 구성 요소 | 기술 |
|-----------|------|
| Backend | FastAPI |
| Templates | Jinja2 |
| 실시간 업데이트 | HTMX (폴링) |
| CSS | Tailwind CSS (CDN) |
| 마크다운 렌더링 | markdown 라이브러리 (서버사이드) |
| DB | SQLite + aiosqlite |
| 백그라운드 작업 | asyncio.create_task (단순), 또는 FastAPI BackgroundTasks |
| OCR | 기존 benchmark/ocr/pipeline.py 재사용 |
| Notion 연동 | notion-client 라이브러리 |

## 의존성

```
fastapi
uvicorn[standard]
jinja2
aiosqlite
python-multipart      # 파일 업로드
markdown              # 마크다운 → HTML 렌더링
Pillow
PyMuPDF
httpx
```

## Notion 연동

### 설정

```env
NOTION_API_KEY=secret_xxx
NOTION_DATABASE_ID=xxx    # 논문을 저장할 Notion 데이터베이스 ID
```

### 동작

- 논문 변환 완료 시 Notion 데이터베이스에 새 페이지 생성
- 페이지 속성: 제목, 날짜, 페이지 수, 상태
- 페이지 본문: 마크다운을 Notion blocks으로 변환
- figure 이미지는 Notion에 업로드 또는 외부 URL 링크

### 수동/자동 선택

- `config.yaml`에서 `auto_sync_notion: true/false` 설정
- 수동: 뷰어 페이지에서 "Notion으로 보내기" 버튼

## 파일 구조

```
paper-pipeline/
├── app.py                # FastAPI 메인 + 라우트
├── models.py             # SQLite ORM (papers 테이블)
├── ocr_worker.py         # 백그라운드 OCR 처리
├── notion_sync.py        # Notion 연동
├── config.py             # 설정 로드
├── config.yaml           # 앱 설정
├── templates/
│   ├── base.html         # 공통 레이아웃 (Tailwind + HTMX)
│   ├── index.html        # 논문 목록 + 업로드 영역
│   ├── viewer.html       # 마크다운 뷰어
│   └── partials/
│       ├── paper_list.html   # 논문 목록 (HTMX partial)
│       ├── upload_status.html # 업로드 진행률 (HTMX partial)
│       └── search_results.html # 검색 결과 (HTMX partial)
├── static/
│   └── style.css         # 추가 CSS
├── data/
│   └── papers/           # 변환된 논문 저장
└── requirements.txt
```

## config.yaml

```yaml
server:
  host: 0.0.0.0
  port: 8080

ocr:
  qwen_base_url: http://localhost:8000/v1
  qwen_model: qwen3.5-27b
  deepseek_url: http://localhost:8003
  dpi: 144
  max_concurrent: 2

storage:
  data_dir: data/papers
  db_path: data/papers.db

notion:
  enabled: false
  api_key: ${NOTION_API_KEY}
  database_id: ${NOTION_DATABASE_ID}
  auto_sync: false
```

## 에러 처리

- **업로드 실패**: 파일 크기 제한 (기본 100MB), PDF 아닌 파일 거부
- **OCR 실패**: status="error" + error_msg 저장, UI에서 재시도 버튼
- **Notion 연동 실패**: 로그 기록, 나중에 수동 재시도 가능
- **서버 재시작**: status="processing"인 문서는 자동 재처리

## 향후 확장 (2단계)

- **요약 생성**: 변환 완료 후 Qwen으로 논문 요약 자동 생성
- **RAG**: 마크다운을 벡터 DB에 인덱싱, 챗 인터페이스로 질의
- **태그/분류**: 논문 주제 자동 분류, 태그 관리
- **협업**: 팀원들과 논문 공유, 코멘트
