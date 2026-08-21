# 🎯 Vision Agent

> A real-time multimodal AI agent that doesn't just listen to your interview answers — it watches you.

**Alex** is an AI interview coach built as a real-time agent, not a chatbot with a video call bolted on. It runs a live video interview, fuses three concurrent signal streams — spoken audio, live video, and structured vision-model output — into one coherent context, reasons over all of it simultaneously, and acts: asking follow-up questions, calling out a phone that just appeared on camera, adjusting tone when eye contact drops, and closing with a scored, spoken debrief the moment the interview ends.

This repo is a case study in **agent orchestration for real-time multimodal systems**: how to keep perception (vision models), reasoning (an LLM), and action (voice, scoring, UI) in sync when everything is happening live, with no room for batch processing or retries.

---

## 🎬 Watch It Work

**[▶ Watch the full demo](https://www.youtube.com/watch?v=mScWgvHX-As)**

[![Watch the Demo](https://img.youtube.com/vi/mScWgvHX-As/0.jpg)](https://www.youtube.com/watch?v=mScWgvHX-As)

*Written breakdown of the build:* [dev.to post](https://dev.to/yamini_priya_4f7873b3baf2/from-zero-to-live-ai-agent-how-i-built-an-interview-coach-with-vision-agents-sdk-58hd)

---

## 🌟 What Alex Does

- **🎙️ Conducts a real interview** — greets you, asks role-specific questions generated from a job description or resume, follows up on vague answers, and paces itself like a human recruiter rather than firing off a fixed script
- **👁️ Reads body language live** — tracks eye contact, posture, and nervousness signals through YOLO pose detection, streamed into the agent's reasoning in real time
- **📱 Monitors the environment** — detects phones, extra screens, or additional people via YOLO object detection and reacts *mid-interview*, not in a post-hoc report
- **💬 Gives spoken, structured feedback** — after the final question, Alex delivers verbal feedback on strengths, gaps, and body language, no waiting for a report to generate
- **📊 Scores the session** — every answer rated on Clarity, Relevance, and Depth, combined with engagement metrics into a final recommendation the moment the call ends

---

## 🏗️ System Architecture

```
Candidate (webcam + mic)
         ↓
    Stream Edge Network (WebRTC, ap-south-1)
         ↓
    Vision Agents SDK  ←── orchestration layer
         ├── YOLOPoseProcessor (yolo11n-pose.pt)
         │       └── eye contact, posture, nervousness → pushed as live state
         ├── YOLOProcessor (yolo11n.pt)
         │       └── phone / extra screens / extra people → pushed as live state
         └── Gemini Realtime (multimodal LLM)
                 └── consumes video + audio + processor state concurrently
                         ↓
              reasons over all three streams → decides next action:
              ask a question, follow up, flag a distraction,
              adjust tone, or move to feedback
                         ↓
         spoken feedback + structured scoring on session end
```

**Why an orchestration SDK instead of hand-rolled glue code:** the alternative to Vision Agents here is manually managing a WebRTC session, polling two YOLO models, buffering their output, and injecting it into an LLM call on some schedule — while also streaming audio and video to that same LLM without breaking sync. Vision Agents collapses this into a processor abstraction: each vision model's output is exposed as *state*, and that state is automatically merged into the LLM's live context on every turn. The agent code only has to define what the processors detect and what the LLM should do with that information — not how the data gets there.

---

## 🧠 Design Decisions & Tradeoffs

**Single realtime multimodal model vs. a pipelined stack (STT → LLM → TTS)**
A traditional pipeline (speech-to-text, then LLM, then text-to-speech) adds latency at every hop and throws away non-verbal signal — tone, hesitation, timing. Gemini Realtime processes audio and video natively and speaks back directly, which is what makes the interview feel like a live conversation instead of a call-and-response bot. The tradeoff: fewer knobs to tune per-stage, and you're locked into a specific model family (few models support bidirectional native audio streaming — see Engineering Challenges below).

**Local YOLO inference vs. a cloud vision API**
Running `yolo11n` and `yolo11n-pose` locally keeps per-frame latency predictable and avoids a second network round-trip stacked on top of the live video call. The cost is that the app is CPU/GPU-bound on whatever machine runs it, which is the main reason this currently runs locally rather than on a thin cloud instance.

**Pushing processor state into LLM context vs. a separate rules engine**
An earlier approach considered hard-coding rules like "if phone detected → say X." Instead, YOLO output is exposed as state and the LLM decides *when and how* to react, based on interview phase and conversational context. This makes the agent's behavior contextual (it won't interrupt mid-sentence to flag a phone) rather than reflexive, at the cost of being less deterministic to test.

**Subclassing processors instead of using default visualization**
The SDK's default `YOLOPoseProcessor`/`YOLOProcessor` draw skeleton overlays on the video feed. For an interview product, showing the user a skeleton is the wrong UX. Every drawing method had to be overridden in a `CleanProcessor` subclass to suppress rendering while keeping detection intact — a reminder that adapting an SDK to a product's actual UX often means overriding more than the docs first suggest.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Agent Runtime / Orchestration** | Vision Agents SDK |
| **Video Infrastructure** | Stream (WebRTC, ap-south-1) |
| **Pose Detection** | YOLO — `yolo11n-pose.pt` |
| **Object Detection** | YOLO — `yolo11n.pt` |
| **Vision + Audio Reasoning** | Google Gemini Realtime (native bidirectional audio) |
| **UI & Report Dashboard** | Streamlit |
| **Language** | Python 3.12 |
| **Package Manager** | UV |

---

## 🐛 Engineering Challenges & How They Were Solved

**Bidirectional audio only works on specific model versions**
Standard Gemini models don't support live, two-way audio streaming. Only `gemini-2.5-flash-native-audio-preview-12-2025` handled it correctly — a reminder that with realtime multimodal APIs, the model version isn't a minor detail, it's a hard capability boundary.

**Suppressing the pose skeleton overlay**
The SDK renders detection results onto the video by default. Fixed by subclassing both YOLO processors and overriding every drawing method — detection logic stays intact, rendering is fully suppressed.

**Reliable phone detection mid-call**
Detection accuracy depended heavily on lighting and phone orientation. Solved with an explicit `ACTIVE VISUAL SCANNING` directive in the agent's dynamic instructions, keeping the object-detection processor's attention weighted correctly during the live conversation rather than treating it as a background check.

**Latency**
Deployed on Stream's `ap-south-1` (Mumbai) edge for lowest latency to the primary user base; this is a config value, not a hard dependency, so it can be repointed for other regions.

**Stuck process state**
Long-running agent processes tracked via `agent.pid` could get orphaned on abnormal exit. Handled with a reset flow in the UI that clears the pid and process state without a manual shell command.

---

## 🎓 Key Learnings (Agent Engineering)

- **Processor state is the real integration point.** The value of an orchestration SDK isn't the API surface — it's that vision-model output becomes something the LLM can reason over natively, without hand-written glue.
- **Instructions are the product.** A well-structured system prompt with explicit phases, priorities, and tone rules is the difference between an agent that feels like a form with a voice and one that feels like a person paying attention.
- **Determinism vs. context-awareness is a real tradeoff.** Letting the LLM decide *when* to act on a signal (rather than hard-coded rules) makes the agent feel natural but makes behavior harder to unit test — worth designing for explicitly, not discovering by accident.
- **Realtime multimodal capability is model-specific, not model-family-generic.** Don't assume "Gemini supports X" — check the exact model version against the exact capability you need.

---

## 🚀 Running Locally

> ⚠️ This project runs locally only. It requires persistent processes, WebRTC connections, and local YOLO inference that most cloud platforms don't support out of the box.

### Prerequisites
- Python 3.12+
- Webcam and microphone
- Google Gemini API key
- Stream API key and secret
- UV package manager

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/Yamini26284/Vision-Agent.git
cd Vision-Agent
```

**2. Install UV if you don't have it**
```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Mac / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**3. Install dependencies**
```bash
uv sync
uv add vision-agents "vision-agents[getstream,gemini,ultralytics]"
```

**4. Create your `.env` file**
```
GEMINI_API_KEY=your_gemini_api_key
STREAM_API_KEY=your_stream_api_key
STREAM_API_SECRET=your_stream_api_secret
```

Get your keys:
- Gemini API key → [aistudio.google.com](https://aistudio.google.com)
- Stream API key → [getstream.io](https://getstream.io)

**5. Run the Streamlit UI**
```bash
streamlit run app.py
```
The UI launches `main.py` automatically in the background when you click **Start My Interview**.

### How to Use
1. Open `http://localhost:8501`
2. Enter your **Target Role** and **Seniority Level**
3. Provide a Job Description, a Resume, or let Alex decide
4. Set the number of questions (1–10)
5. Click **🎙️ Start My Interview with Alex**
6. Say **Hello** to begin
7. Complete the interview and debrief with Alex
8. Click **Report** in the sidebar for the full performance breakdown

---

## 📁 Project Structure

```
Vision-Agent/
├── .venv/
├── .env                      # API keys (not committed)
├── .gitignore
├── .python-version
├── agent.pid                 # auto-generated, tracks the running agent process
├── app.py                    # Streamlit UI — setup form + live call embed
├── interview_config.json     # written by app.py, read by main.py
├── main.py                   # Vision Agents agent — interview logic and lifecycle
├── packages.txt
├── pyproject.toml            # UV dependencies
├── README.md
├── requirements.txt
├── uv.lock
├── yolo11n-pose.pt           # YOLO pose model
├── yolo11n.pt                # YOLO object detection model
└── yolo26n-pose.pt           # YOLO pose model (larger variant)
```

---

## 🔮 Future Plans

- [ ] Session history and progress tracking over time
- [ ] Answer content analysis beyond body language signals
- [ ] Multiple interview modes — technical, behavioral, case study
- [ ] Cloud deployment (Railway + Supabase)
- [ ] Mobile support
- [ ] Multi-language interviews

---

## 🙏 Acknowledgments

- [Vision Agents SDK](https://github.com/GetStream/vision-agents) by Stream — the orchestration framework this agent is built on
- [Google Gemini](https://ai.google.dev/) — real-time multimodal reasoning
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — pose and object detection
- [Stream](https://getstream.io) — live video infrastructure

---

## 👤 Author

**Yamini Priya**
- GitHub: [@Yamini26284](https://github.com/Yamini26284)
- LinkedIn: [yamini26284](https://www.linkedin.com/in/yamini26284)

---

## ⭐ Support

If this project is useful or interesting to you, a star helps it reach more people.
