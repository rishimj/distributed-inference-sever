# 🎬 Demo Guide - Distributed Inference Server

This guide shows you how to demo the distributed inference server **for free** without requiring GPUs.

## 🆓 Free Demo Options

### Option 1: Local Demo (Recommended) ⭐

**Requirements**: Docker & Docker Compose (no GPU needed!)

**Start the demo:**
```bash
./scripts/start_demo.sh
```

**Access the demo:**
- 🌐 **Web UI**: http://localhost:8080 - Interactive demo interface
- 🔌 **API**: http://localhost:8000 - REST API endpoint
- 📊 **Grafana**: http://localhost:3000 - Dashboards (admin/demo123)
- 📈 **Prometheus**: http://localhost:9090 - Metrics

**Features**:
- ✅ Works without GPU (uses mock workers)
- ✅ Full system architecture running
- ✅ Real routing and cache logic
- ✅ Live metrics and monitoring
- ✅ Interactive web interface

**Stop the demo:**
```bash
docker-compose -f docker-compose.demo.yml down
```

---

### Option 2: GitHub Codespaces

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=YOUR_REPO_ID)

**Steps:**
1. Click "Open in GitHub Codespaces" badge
2. Wait for environment to start
3. Run: `./scripts/start_demo.sh`
4. Access forwarded ports (Codespaces will auto-forward)

**Cost**: Free tier includes 60 hours/month

---

### Option 3: Gitpod

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/rishimj/distributed-inference-sever)

**Steps:**
1. Click "Open in Gitpod" badge
2. Run: `./scripts/start_demo.sh`
3. Click on forwarded port 8080

**Cost**: Free tier includes 50 hours/month

---

### Option 4: Play with Docker

Visit: https://labs.play-with-docker.com/

**Steps:**
1. Click "Start"
2. Add new instance
3. Clone repo: `git clone https://github.com/rishimj/distributed-inference-sever.git`
4. `cd distributed-inference-sever`
5. `./scripts/start_demo.sh`
6. Click on exposed port links

**Cost**: Completely free, no signup needed
**Limitation**: 4-hour sessions

---

### Option 5: Video Demo

Can't run locally? Watch the demo video:

📺 **[Watch Demo Video](https://youtu.be/YOUR_VIDEO_ID)** (Coming soon)

Shows:
- System architecture
- Cache-aware routing in action
- Performance metrics
- Multi-worker load balancing

---

### Option 6: Interactive Documentation

GitHub Pages demo site with:
- Architecture diagrams
- API documentation
- Example requests/responses
- Performance benchmarks

🔗 **[View Interactive Docs](https://rishimj.github.io/distributed-inference-sever/)** (Coming soon)

---

## 📸 Screenshots & Demos

### Web Interface
![Demo UI](./demo/screenshots/demo-ui.png)

### Grafana Dashboard
![Grafana](./demo/screenshots/grafana-dashboard.png)

### Metrics
![Metrics](./demo/screenshots/prometheus-metrics.png)

---

## 🧪 Testing the Demo

### 1. Health Check
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "workers": 2,
  "uptime": "5m 32s"
}
```

### 2. Generate Text
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

### 3. Check Metrics
```bash
curl http://localhost:8000/metrics
```

### 4. Worker Status
```bash
curl http://localhost:8000/workers
```

---

## 🎥 Creating Your Own Demo Video

Want to showcase your deployment? Here's how:

### Using OBS Studio (Free)

1. **Install OBS**: https://obsproject.com/
2. **Record the demo**:
   - Start with architecture diagram
   - Show the web UI
   - Make inference requests
   - Show metrics updating
   - Display Grafana dashboards

3. **Upload to YouTube/Vimeo**:
   - Title: "Distributed Inference Server Demo"
   - Add timestamps in description
   - Link to GitHub repo

### Using Asciinema (Terminal Recording)

```bash
# Install
pip install asciinema

# Record
asciinema rec demo.cast

# Upload
asciinema upload demo.cast
```

---

## 🌐 Free Cloud Hosting Options

### Railway.app
- Free $5/month credit
- Supports Docker
- Easy deployment
- **Limitation**: No GPU support, use demo mode

### Render.com
- Free tier available
- Docker support
- Auto-deploy from GitHub
- **Limitation**: No GPU, use demo mode

### Heroku
- Free dyno hours available
- Container deployment
- **Limitation**: No GPU, limited to demo mode

### Fly.io
- Free tier includes 3 VMs
- Docker deployment
- **Limitation**: No GPU, demo mode only

---

## 📊 Real GPU Demo

For showcasing with real vLLM on GPU:

### Free GPU Options

1. **Google Colab** (Free tier includes GPU)
```bash
# In Colab notebook
!git clone https://github.com/rishimj/distributed-inference-sever.git
%cd distributed-inference-sever
!./scripts/start_system.sh
```

2. **Kaggle Notebooks** (Free GPU: 30h/week)
- Similar to Colab
- Better GPU quota

3. **Lambda Labs** (Free trial credits)
- Real GPU instances
- Full vLLM support

4. **Vast.ai** (Low-cost GPU rental)
- Pay-as-you-go
- Starting at $0.10/hour

---

## 🎯 Best Practices for Demos

### For GitHub README
1. Add badges at the top
2. Include screenshots
3. Add quick start section
4. Link to live demo
5. Embed demo video

### For Presentations
1. Start with architecture diagram
2. Show working system
3. Demonstrate cache-aware routing
4. Show metrics and monitoring
5. Discuss performance benefits

### For Portfolio
1. Create dedicated demo page
2. Add technical writeup
3. Include performance benchmarks
4. Show scaling capabilities
5. Discuss design decisions

---

## 🆘 Troubleshooting Demo

### Ports Already in Use
```bash
# Check what's using the port
lsof -i :8000

# Use different ports
export GATEWAY_PORT=8100
export WORKER1_PORT=8101
export WORKER2_PORT=8102
```

### Docker Issues
```bash
# Clean up
docker-compose -f docker-compose.demo.yml down -v

# Rebuild
docker-compose -f docker-compose.demo.yml build --no-cache

# Restart
docker-compose -f docker-compose.demo.yml up -d
```

### Can't Access UI
```bash
# Check services are running
docker-compose -f docker-compose.demo.yml ps

# Check logs
docker-compose -f docker-compose.demo.yml logs gateway

# Verify connectivity
curl http://localhost:8000/health
```

---

## 📝 Demo Checklist

- [ ] Clone repository
- [ ] Run `./scripts/start_demo.sh`
- [ ] Access web UI at http://localhost:8080
- [ ] Test inference request
- [ ] Check metrics in Prometheus
- [ ] View dashboards in Grafana
- [ ] Test cache-aware routing
- [ ] Monitor worker health
- [ ] Stop demo cleanly

---

## 🎉 Share Your Demo

Share your demo on:
- Twitter: #DistributedInference #vLLM #LLM
- LinkedIn: Tag relevant people/companies
- Reddit: r/MachineLearning, r/LocalLLaMA
- Hacker News: Show HN thread

**Demo link template**:
```
🚀 Built a distributed LLM inference server with:
- Cache-aware routing
- Horizontal scaling
- Real-time metrics
- Production-ready

Try the demo: [your-demo-url]
GitHub: https://github.com/rishimj/distributed-inference-sever
```

---

## 🤝 Contributing Demos

Improved the demo? Submit a PR!

Ideas for demo improvements:
- [ ] Add more interactive visualizations
- [ ] Real-time cache hit rate graph
- [ ] Worker load distribution chart
- [ ] Request flow animation
- [ ] Performance comparison tool
- [ ] Load testing dashboard

---

**Questions?** Open an issue on GitHub!
