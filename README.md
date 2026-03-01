# 🎯 Vision Agent

> An AI interviewer that doesn't just listen to your answers. It watches you.

Built for students and job seekers who can't afford coaching, Alex conducts real-time video interviews, tracks your body language through pose detection, catches distractions in your environment, and delivers honest verbal feedback the moment you're done — all for free.


[🎥 Watch Demo](https://www.youtube.com/watch?v=mScWgvHX-As) | [📝 Read Blog Post](https://dev.to/yamini_priya_4f7873b3baf2/from-zero-to-live-ai-agent-how-i-built-an-interview-coach-with-vision-agents-sdk-58hd) | [💻 GitHub](https://github.com/Yamini26284)

---

## 🌟 What Alex Does

- **🎙️ Conducts real interviews** — greets you, asks role-specific questions, follows up on vague answers, and waits patiently like a real recruiter
- **👁️ Watches your body language** — tracks eye contact, posture, and nervousness levels through YOLO pose detection in real time
- **📱 Monitors your environment** — detects mobile phones, extra screens, and additional people using YOLO object detection and reacts mid-interview
- **💬 Gives live verbal feedback** — after the final question, Alex delivers structured feedback on your strengths, areas to improve, and body language
- **📊 Generates a performance report** — every answer scored on Clarity, Relevance and Depth, with engagement metrics and a final recommendation on screen the moment you say goodbye

---

## 🎬 Demo

[![Watch the Demo](https://img.youtube.com/vi/mScWgvHX-As/0.jpg)](https://www.youtube.com/watch?v=mScWgvHX-As)

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Agent Runtime** | Vision Agents SDK |
| **Video Infrastructure** | Stream (ap-south-1, Mumbai) |
| **Pose Detection** | YOLO yolo11n-pose.pt |
| **Object Detection** | YOLO yolo11n.pt |
| **Vision + Audio LLM** | Google Gemini Realtime |
| **UI & Report Dashboard** | Streamlit |
| **Language** | Python 3.12 |
| **Package Manager** | UV |

---

## 🏗️ Architecture

```
Candidate (webcam + mic)
         ↓
    Stream Edge Network (ap-south-1)
         ↓
    Vision Agents SDK  ←── Orchestrates everything
         ├── YOLOPoseProcessor (yolo11n-pose.pt)
         │       └── Eye contact, posture, nervousness → passed as state to Gemini
         ├── YOLOPoseProcessor (yolo11n.pt)
         │       └── Phone, extra screens, people detected → passed as state to Gemini
         └── Gemini Realtime (VideoLLM)
                 └── Sees video + hears audio simultaneously
                         ↓
              Reacts in real time — asks questions,
              follows up, addresses distractions,
              monitors body language
                         ↓
         gives real time feedback, suggestions to improve and room for more discussions.
```

**Why Vision Agents SDK?**
Vision Agents is the orchestration layer that makes all of this possible. Without it, wiring YOLO pose detection, YOLO object detection, and Gemini Realtime together with a live video call would take weeks. Vision Agents handles the entire pipeline — processor state flows automatically into the LLM's context, so Gemini knows what YOLO sees without any manual piping.

---

## 🚀 Running Locally

> ⚠️ This project runs locally only. It requires persistent processes, WebRTC connections, and YOLO inference that cloud platforms do not support.

### Prerequisites

- Python 3.12+
- Webcam and microphone
- Google Gemini API key
- Stream API key and secret
- UV package manager

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/Yamini26284/ai-interview-coach.git
cd ai-interview-coach
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
```env
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

That's it. The UI launches `main.py` automatically in the background when you click **Start My Interview**.

---

## 📖 How to Use

1. Open `http://localhost:8501` in your browser
2. Enter your **Target Role** and **Seniority Level**
3. Choose how Alex should prepare — paste a Job Description, paste your Resume, or let AI decide
4. Set the number of questions (1–10)
5. Click **🎙️ Start My Interview with Alex**
6. The interview call opens automatically — just say **Hello** to begin
7. Complete your interview and debrief with Alex
8. Click **Report** in the sidebar to view your full performance analysis

---

## 📁 Project Structure

```
ai-interview-coach/
├── .venv/
├── .env                      # API keys (not committed)
├── .gitignore
├── .python-version
├── agent.pid                 # Auto-generated, tracks running agent process
├── app.py                    # Streamlit UI — setup form and live call embed
├── interview_config.json     # Written by app.py, read by main.py
├── main.py                   # Vision Agents agent — interview logic and lifecycle
├── packages.txt
├── pyproject.toml            # UV dependencies
├── README.md
├── requirements.txt
├── uv.lock
├── yolo11n-pose.pt           # YOLO pose model
├── yolo11n.pt                # YOLO object detection model
└── yolo26n-pose.pt           # YOLO pose model (larger)
```

---

## 🐛 Troubleshooting

**Agent not speaking / silent on call**
- Make sure you're using `gemini-2.5-flash-native-audio-preview-12-2025` — standard Gemini models don't support bidirectional audio streaming

**Skeleton lines visible on video**
- The `CleanProcessor` subclass in `main.py` suppresses drawing — make sure it's applied to both YOLO processors

**Phone not detected mid-interview**
- Ensure `yolo11n.pt` is present in the project root
- Hold the phone clearly visible, front-facing in good lighting
- The `ACTIVE VISUAL SCANNING` instruction in `dynamic_instructions` must be present

**High latency**
- You're on `ap-south-1` (Mumbai) by default — best for India
- Close other bandwidth-heavy applications
- Check your internet connection speed

**`agent.pid` file stuck / can't start new session**
- Click **Reset & Start Fresh** on the home page
- Or manually delete `agent.pid` from the project folder

---

## 🎓 Key Learnings

- **Processor state is the bridge** — Vision Agents passes YOLO output directly into Gemini's context. Understanding this unlocks the full power of the SDK
- **Instructions are the product** — a well-structured prompt with phases, priorities and tone rules is the difference between an agent that feels human and one that feels like a form with a voice
- **Subclass when the SDK doesn't expose what you need** — overriding every drawing method on `YOLOPoseProcessor` was the only way to cleanly suppress the skeleton overlay
- **Realtime models are specific** — not all Gemini models support native audio streaming. Model version matters enormously

---

## 🔮 Future Plans

- [ ] Session history and progress tracking over time
- [ ] Answer content analysis beyond body language
- [ ] Multiple interview modes — technical, behavioural, case study
- [ ] Cloud deployment via Railway + Supabase
- [ ] Mobile support
- [ ] Multi-language interviews

---

## 🙏 Acknowledgments

- [Vision Agents SDK](https://github.com/GetStream/vision-agents) by Stream — the framework that made this possible in 4 days
- [Google Gemini](https://ai.google.dev/) — for real-time multimodal understanding
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — for pose and object detection
- [Stream](https://getstream.io) — for live video infrastructure
- Built for [Vision Possible: Agent Protocol — WeMakeDevs Hackathon 2026](https://www.wemakedevs.org)

---

## 👤 Author

**Yamini Priya**
- GitHub: [@Yamini26284](https://github.com/Yamini26284)
- LinkedIn: [yamini26284](https://www.linkedin.com/in/yamini26284)

---

## ⭐ Support

If this project helped you or inspired you, give it a ⭐ — it means a lot.