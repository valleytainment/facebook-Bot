# facebook-Bot
A feature-rich Facebook automation bot with GPT-4 integration, encrypted credential storage, scheduling, sentiment analysis, media uploads, analytics, and more. Developed for educational/demo use.
# Facebook Bot Master Overlord

**A fully featured Facebook automation bot with GPT-4 integration, encrypted credentials, scheduling, sentiment analysis, media uploads, analytics, and more — all wrapped in a Python/Tkinter GUI.**

---

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation and Setup](#installation-and-setup)
4. [Usage](#usage)
5. [Configuration](#configuration)
6. [Security & Privacy](#security--privacy)
7. [Limitations & Warnings](#limitations--warnings)
8. [Troubleshooting](#troubleshooting)
9. [Contributing](#contributing)
10. [License](#license)

---

## Introduction

This project, **Facebook Bot Master Overlord**, provides a **powerful yet easy-to-use** tool for automating Facebook posts and comments, complete with:

- **A robust GUI** built in Python/Tkinter
- **Advanced AI content generation** (GPT-4)
- **Encrypted credential storage** with AES-256
- **Scheduling & automation** using APScheduler
- **Sentiment & compliance checks** via NLTK
- **Media upload & preview** support
- **Analytics dashboard** for engagement tracking

**Disclaimer:** This project is provided for **educational/demo purposes** only. Automated posting or commenting may violate Facebook’s Terms of Service. **Use responsibly.**

---

## Features

1. **Encrypted Credentials**  
   - Uses AES-256 via `cryptography`’s Fernet to store Facebook and GPT API keys securely.

2. **GPT-4 Integration**  
   - Generate engaging posts or comments with the OpenAI GPT API.  
   - Fallback messages included if GPT is unavailable.

3. **Sentiment & Compliance**  
   - NLTK’s VADER for sentiment analysis and a real-time “bad words” list for content compliance/sanitization.

4. **Media Support**  
   - Preview local image/video files and prepare them for posting (though posting media to Facebook might require additional Graph API endpoints).

5. **Scheduling & Automation**  
   - APScheduler for auto-posting and auto-commenting at custom intervals.

6. **Analytics Dashboard**  
   - Real-time engagement tracking with a Matplotlib chart.

7. **Two-Factor Authentication (TOTP)**  
   - Optional 2FA with `pyotp` to further secure your bot’s usage.

8. **Proxy Support**  
   - Route traffic through a configurable proxy.

---

## Installation and Setup

1. **Clone or Download** this repository:
   ```bash
   git clone https://github.com/<your-username>/facebook-bot-master-overlord.git
   cd facebook-bot-master-overlord
