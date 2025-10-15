@echo off
REM --- detection strictness (words) ---
set WORD_OPEN_FRAC_MIN=0.65
set WORD_STABLE_FRAMES=4
set WORD_VOTES_K=6
set WORD_MIN_SCORE=0.80
REM Lowered from 0.78 for more detections
set WORD_MIN_AREA=0.075
REM Lowered from 0.08 for smaller hands
set WORD_MOTION_STD_MIN=0.0015
set WORD_MOTION_STD_MAX=0.010
set WORD_HAND_STREAK_ON=2
REM Increased from 3 for more stability
set WORD_SKIN_MIN_FRAC=0.08
set WORD_COOLDOWN_S=6.0
set WORD_ML_COOLDOWN_S=6.0
set BUFFER_IDLE_CLEAR_S=4.0
REM Up from 2.5 to reduce clears
set WORD_DEBUG=2
REM Set to 1 for candidate logs (less verbose than 2)
set MOTION_STD_VETO=0.016
set LETTER_SPEAK_MAX_MOTION=0.030
REM Increased from 0.025 to allow slight motion

set WAVE_MIN_FLIPS=6
set WAVE_MIN_AMP=0.14  
REM Example: Raise from default ~0.07 to filter weak waves
set q=5   
REM For 'hi' (was 4); test for specificity

REM --- letters ---
set INFER_EVERY=1
set LETTER_MAJORITY_WINDOW=12
REM Increased from 15 for better majority voting

REM --- mediapipe/tf logging ---
set TF_CPP_MIN_LOG_LEVEL=2
set GLOG_minloglevel=2
set GLOG_logtostderr=1

REM --- optional perf caps (avoid CPU thrash) ---
set OMP_NUM_THREADS=2
set TF_NUM_INTRAOP_THREADS=2
set TF_NUM_INTEROP_THREADS=1

REM --- camera & MP stability (optional) ---
set CAM_INDEX=0
set MP_MODEL_COMPLEXITY=1
set MP_MIN_DET_CONF=0.50
REM Lowered from 0.60 for better hand pickup
set MP_MIN_TRK_CONF=0.50
REM Lowered from 0.60

REM --- run ---
python ui\app_hybrid.py