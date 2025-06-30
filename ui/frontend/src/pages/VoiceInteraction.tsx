import React, { useState, useEffect, useRef } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Mic, MicOff, Volume2, VolumeX, ArrowLeft, Zap, Loader, Settings, X } from 'lucide-react';
import { API_BASE_URL, API_ENDPOINTS } from '../config/api';

// Speech Recognition types
interface SpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList;
  resultIndex: number;
}

interface SpeechRecognitionResultList {
  length: number;
  item(index: number): SpeechRecognitionResult;
  [index: number]: SpeechRecognitionResult;
}

interface SpeechRecognitionResult {
  length: number;
  item(index: number): SpeechRecognitionAlternative;
  [index: number]: SpeechRecognitionAlternative;
  isFinal: boolean;
}

interface SpeechRecognitionAlternative {
  transcript: string;
  confidence: number;
}

interface SpeechRecognitionErrorEvent extends Event {
  error: string;
  message: string;
}

interface SpeechRecognition extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  maxAlternatives: number;
  serviceURI: string;
  grammars: any;
  start(): void;
  stop(): void;
  abort(): void;
  onstart: ((this: SpeechRecognition, ev: Event) => any) | null;
  onend: ((this: SpeechRecognition, ev: Event) => any) | null;
  onerror: ((this: SpeechRecognition, ev: SpeechRecognitionErrorEvent) => any) | null;
  onresult: ((this: SpeechRecognition, ev: SpeechRecognitionEvent) => any) | null;
}

declare global {
  interface Window {
    SpeechRecognition: {
      new(): SpeechRecognition;
    };
    webkitSpeechRecognition: {
      new(): SpeechRecognition;
    };
  }
}

interface VoiceInteractionProps {}

const VoiceInteraction: React.FC<VoiceInteractionProps> = () => {
  const navigate = useNavigate();
  const location = useLocation();
  
  // Character data from navigation state
  const characterId = location.state?.characterId;
  const characterName = location.state?.characterName || "Character";
  const characterImage = location.state?.characterImage;
  
  // Recording state
  const [isListening, setIsListening] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [isGeneratingResponse, setIsGeneratingResponse] = useState(false);
  const [isPlayingResponse, setIsPlayingResponse] = useState(false);
  
  // Audio state
  const [audioLevel, setAudioLevel] = useState(0);
  const [transcript, setTranscript] = useState('');
  const [lastResponse, setLastResponse] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [speakingWaveform, setSpeakingWaveform] = useState<number[]>(Array(12).fill(0));
  
  // Settings
  const [volumeThreshold, setVolumeThreshold] = useState(0.01); // Lower threshold for better sensitivity
  const [silenceDuration, setSilenceDuration] = useState(4.0); // Increased from 2.0 to 4.0 seconds
  const [autoPlayEnabled, setAutoPlayEnabled] = useState(true);
  const [showSettings, setShowSettings] = useState(false);
  
  // Refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const silenceTimerRef = useRef<number | null>(null);
  const maxRecordingTimerRef = useRef<number | null>(null);
  const lastSpeechTimeRef = useRef<number>(0);
  const currentAudioRef = useRef<HTMLAudioElement | null>(null);
  
  // Audio feedback for recording start
  const playRecordingStartSound = () => {
    // Create a short beep sound using Web Audio API
    if (audioContextRef.current) {
      try {
        const oscillator = audioContextRef.current.createOscillator();
        const gainNode = audioContextRef.current.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContextRef.current.destination);
        
        oscillator.frequency.setValueAtTime(800, audioContextRef.current.currentTime); // 800Hz beep
        gainNode.gain.setValueAtTime(0.1, audioContextRef.current.currentTime); // Low volume
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContextRef.current.currentTime + 0.1);
        
        oscillator.start(audioContextRef.current.currentTime);
        oscillator.stop(audioContextRef.current.currentTime + 0.1); // 100ms beep
        
        console.log('🔊 Played recording start sound');
      } catch (error) {
        console.error('Failed to play recording start sound:', error);
      }
    }
  };
  
  // Animation frame for audio level monitoring
  const animationFrameRef = useRef<number | null>(null);
  const waveformAnimationRef = useRef<number | null>(null);
  
  // Wake word detection
  const [isWakeWordListening, setIsWakeWordListening] = useState(false);
  const speechRecognitionRef = useRef<SpeechRecognition | null>(null);
  const wakeWordTimeoutRef = useRef<number | null>(null);
  const isProcessingWakeWordRef = useRef<boolean>(false);
  const restartTimeoutRef = useRef<number | null>(null);

  useEffect(() => {
    // Redirect if no character selected
    if (!characterId) {
      navigate('/character-selection');
      return;
    }

    // Initialize audio monitoring
    initializeAudioMonitoring();
    
    return () => {
      cleanup();
    };
  }, [characterId, navigate]);

  // Initialize wake word detection when listening starts
  useEffect(() => {
    if (isListening && !isWakeWordListening) {
      startWakeWordDetection();
    } else if (!isListening && isWakeWordListening) {
      stopWakeWordDetection();
    }
  }, [isListening]);

  const startWakeWordDetection = () => {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      console.warn('Speech recognition not supported in this browser');
      return;
    }

    // Prevent multiple instances
    if (speechRecognitionRef.current || isWakeWordListening) {
      console.log('⚠️ Wake word detection already running or starting');
      return;
    }

    try {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      const recognition = new SpeechRecognition();
      
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-US';
      recognition.maxAlternatives = 1;

      recognition.onstart = () => {
        console.log('🎯 Wake word detection started');
        setIsWakeWordListening(true);
      };

      recognition.onresult = (event: SpeechRecognitionEvent) => {
        // Skip if we're already processing a wake word
        if (isProcessingWakeWordRef.current) return;
        
        const lastResult = event.results[event.results.length - 1];
        if (lastResult && lastResult[0] && lastResult.isFinal) {
          const transcript = lastResult[0].transcript.toLowerCase().trim();
          console.log('🎤 Heard (final):', transcript);
          
          // Check for wake word variations - only check final results
          const wakeWords = [
            'hey oppenheimer',
            'hi oppenheimer', 
            'hello oppenheimer'
          ];
          
          const containsWakeWord = wakeWords.some(wakeWord => 
            transcript.includes(wakeWord)
          );
          
          if (containsWakeWord && !isRecording && !isTranscribing && !isGeneratingResponse && !isProcessingWakeWordRef.current) {
            console.log('🚀 Wake word detected! Starting recording...');
            
            // Set processing flag to prevent multiple triggers
            isProcessingWakeWordRef.current = true;
            
            // Stop wake word detection immediately
            recognition.stop();
            
            // Start recording after a brief delay to ensure wake word detection has fully stopped
            setTimeout(async () => {
              if (!isRecording && !isTranscribing && !isGeneratingResponse) {
                console.log('🎤 Starting recording after wake word detection...');
                await startRecording();
              }
              // Reset processing flag after a delay
              setTimeout(() => {
                isProcessingWakeWordRef.current = false;
                console.log('🔄 Processing flag reset - ready for next wake word');
              }, 3000);
            }, 800); // Longer delay to ensure audio stream is available
          }
        }
      };

      recognition.onerror = (event) => {
        console.error('Wake word detection error:', event);
        setIsWakeWordListening(false);
        speechRecognitionRef.current = null;
        
        // Only restart if it's not an abort error (which happens during normal stop)
        if (event.error !== 'aborted' && isListening && !isRecording && !isTranscribing && !isGeneratingResponse) {
          console.log('🔄 Restarting wake word detection after error...');
          setTimeout(() => {
            if (isListening && !isRecording && !isTranscribing && !isGeneratingResponse && !speechRecognitionRef.current) {
              startWakeWordDetection();
            }
          }, 3000);
        }
      };

      recognition.onend = () => {
        console.log('🎯 Wake word detection ended');
        setIsWakeWordListening(false);
        speechRecognitionRef.current = null;
        
        // Only restart if we're not processing a wake word and still supposed to be listening
        // And only if we haven't been stopped intentionally
        console.log('🔍 Checking restart conditions:', {
          isListening,
          isRecording,
          isTranscribing,
          isGeneratingResponse,
          isProcessingWakeWord: isProcessingWakeWordRef.current,
          hasRestartTimeout: !!restartTimeoutRef.current,
          hasSpeechRecognition: !!speechRecognitionRef.current
        });
        
        if (isListening && !isRecording && !isTranscribing && !isGeneratingResponse && !restartTimeoutRef.current) {
          console.log('🔄 Scheduling wake word detection restart...');
          restartTimeoutRef.current = window.setTimeout(() => {
            restartTimeoutRef.current = null;
            // Reset processing flag and restart
            isProcessingWakeWordRef.current = false;
            if (isListening && !speechRecognitionRef.current) {
              console.log('🚀 Restarting wake word detection after timeout');
              startWakeWordDetection();
            } else {
              console.log('⚠️ Skipping restart - conditions changed');
            }
          }, 2000); // Reduced delay
        } else {
          console.log('⚠️ Not restarting wake word detection - conditions not met');
        }
      };

      speechRecognitionRef.current = recognition;
      recognition.start();
      
    } catch (error) {
      console.error('Failed to start wake word detection:', error);
    }
  };

  const stopWakeWordDetection = () => {
    if (speechRecognitionRef.current) {
      speechRecognitionRef.current.stop();
      speechRecognitionRef.current = null;
    }
    if (wakeWordTimeoutRef.current) {
      clearTimeout(wakeWordTimeoutRef.current);
      wakeWordTimeoutRef.current = null;
    }
    if (restartTimeoutRef.current) {
      clearTimeout(restartTimeoutRef.current);
      restartTimeoutRef.current = null;
    }
    setIsWakeWordListening(false);
    isProcessingWakeWordRef.current = false;
  };

  const cleanup = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    if (waveformAnimationRef.current) {
      cancelAnimationFrame(waveformAnimationRef.current);
    }
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
    }
    if (maxRecordingTimerRef.current) {
      clearTimeout(maxRecordingTimerRef.current);
    }
    if (wakeWordTimeoutRef.current) {
      clearTimeout(wakeWordTimeoutRef.current);
    }
    if (restartTimeoutRef.current) {
      clearTimeout(restartTimeoutRef.current);
    }
    if (speechRecognitionRef.current) {
      speechRecognitionRef.current.stop();
    }
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current.currentTime = 0;
    }
    // Reset processing flags
    isProcessingWakeWordRef.current = false;
  };

  const initializeAudioMonitoring = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000
        } 
      });
      
      streamRef.current = stream;
      
      // Set up audio context for volume monitoring
      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;
      
      const analyser = audioContext.createAnalyser();
      analyserRef.current = analyser;
      analyser.fftSize = 256;
      
      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);
      
      // Set up MediaRecorder
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      mediaRecorderRef.current = mediaRecorder;
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = () => {
        processRecording();
      };
      
      setIsListening(true);
      startVolumeMonitoring();
      
    } catch (err) {
      setError('Failed to access microphone. Please allow microphone access.');
      console.error('Error accessing microphone:', err);
    }
  };

  const startVolumeMonitoring = () => {
    if (!analyserRef.current) return;
    
    const analyser = analyserRef.current;
    const dataArray = new Uint8Array(analyser.frequencyBinCount);
    
    const monitor = () => {
      if (!isListening || isTranscribing || isGeneratingResponse) {
        animationFrameRef.current = requestAnimationFrame(monitor);
        return;
      }
      
      // Check audio context state
      if (audioContextRef.current && audioContextRef.current.state !== 'running' && isRecording) {
        console.log('⚠️ Audio context not running during recording:', audioContextRef.current.state);
      }
      
      analyser.getByteFrequencyData(dataArray);
      
      // Calculate RMS volume
      let sum = 0;
      for (let i = 0; i < dataArray.length; i++) {
        sum += (dataArray[i] / 255) ** 2;
      }
      const rms = Math.sqrt(sum / dataArray.length);
      setAudioLevel(rms);
      
      // Voice activity detection for recording stop (when recording is active)
      if (isRecording && mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
        const voiceDetected = rms > volumeThreshold;
        
        // Always log volume data when recording to debug the issue
        console.log(`🔊 Volume: ${rms.toFixed(4)}, Threshold: ${volumeThreshold}, Voice: ${voiceDetected}`);
        
        if (!voiceDetected) {
          // Start silence timer
          if (!silenceTimerRef.current) {
            console.log(`⏰ Starting silence timer (${silenceDuration}s) at ${new Date().toISOString()}`);
            const startTime = Date.now();
            silenceTimerRef.current = window.setTimeout(() => {
              const elapsed = Date.now() - startTime;
              console.log(`🔇 Silence timer fired after ${elapsed}ms - stopping recording`);
              if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
                stopRecording();
              }
            }, silenceDuration * 1000);
          }
        } else if (voiceDetected && silenceTimerRef.current) {
          // Cancel silence timer if voice detected again
          console.log('🗣️ Voice detected again - canceling silence timer');
          clearTimeout(silenceTimerRef.current);
          silenceTimerRef.current = null;
          // Update last speech time for backup detection
          lastSpeechTimeRef.current = Date.now();
        }
        
        // Update backup silence detection when any volume is detected
        if (rms > 0.005) { // Slightly higher threshold to avoid false positives from background noise
          lastSpeechTimeRef.current = Date.now();
          console.log('🗣️ Speech activity detected - resetting silence timer');
        }
      }
      
      animationFrameRef.current = requestAnimationFrame(monitor);
    };
    
    monitor();
  };

  const startRecording = async () => {
    if (!mediaRecorderRef.current || isRecording || isProcessing || isTranscribing || isGeneratingResponse) {
      console.log('⚠️ Cannot start recording - invalid state');
      return;
    }
    
    // Check MediaRecorder state
    if (mediaRecorderRef.current.state !== 'inactive') {
      console.log('⚠️ MediaRecorder not in inactive state:', mediaRecorderRef.current.state);
      return;
    }
    
    // Check if audio stream is still active
    if (streamRef.current) {
      const tracks = streamRef.current.getAudioTracks();
      console.log('🎵 Audio tracks status:', tracks.map(track => ({
        id: track.id,
        enabled: track.enabled,
        readyState: track.readyState,
        muted: track.muted
      })));
      
      if (tracks.length === 0 || tracks.some(track => track.readyState === 'ended')) {
        console.log('⚠️ Audio stream not available - reinitializing...');
        await initializeAudioMonitoring();
        // Try again after reinitialization
        setTimeout(() => {
          if (!isRecording) {
            startRecording();
          }
        }, 500);
        return;
      }
    }
    
    console.log('🎙️ Starting recording...');
    audioChunksRef.current = [];
    setIsRecording(true);
    setTranscript('');
    setError(null);
    
    // Ensure audio context is running
    if (audioContextRef.current && audioContextRef.current.state === 'suspended') {
      console.log('🔊 Resuming suspended audio context...');
      try {
        await audioContextRef.current.resume();
        console.log('✅ Audio context resumed');
      } catch (error) {
        console.error('Failed to resume audio context:', error);
      }
    }
    
    // Completely rebuild the audio analyser to fix volume monitoring
    if (streamRef.current && audioContextRef.current) {
      try {
        // Create a new analyser
        const newAnalyser = audioContextRef.current.createAnalyser();
        newAnalyser.fftSize = 256;
        
        // Create a new source and connect it
        const source = audioContextRef.current.createMediaStreamSource(streamRef.current);
        source.connect(newAnalyser);
        
        // Replace the old analyser
        analyserRef.current = newAnalyser;
        
        console.log('🔊 Created new audio analyser for volume monitoring');
        
        // Test the analyser immediately
        const testData = new Uint8Array(newAnalyser.frequencyBinCount);
        newAnalyser.getByteFrequencyData(testData);
        const testSum = testData.reduce((a, b) => a + b, 0);
        console.log('🔊 Analyser test - data length:', testData.length, 'sum:', testSum);
        
      } catch (error) {
        console.error('Failed to create new analyser:', error);
      }
    }
    
    try {
      mediaRecorderRef.current.start();
      console.log('✅ Recording started successfully');
      
      // Play audio feedback to indicate recording has started
      playRecordingStartSound();
      
      // Set maximum recording time as fallback (20 seconds - increased for longer responses)
      const maxRecordingStartTime = Date.now();
      console.log('⏰ Setting max recording timer for 20 seconds at', new Date().toISOString());
      
              const maxRecordingCallback = () => {
          const elapsed = Date.now() - maxRecordingStartTime;
          console.log(`⏰ Max recording timer fired after ${elapsed}ms (expected 20000ms)`);
          if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
            console.log('🛑 Forcing stop due to max time limit');
            stopRecording();
          } else {
            console.log('⚠️ MediaRecorder not in recording state:', mediaRecorderRef.current?.state);
          }
        };
        
        maxRecordingTimerRef.current = window.setTimeout(maxRecordingCallback, 20000);
      
              // Also start a backup silence detection that doesn't rely on volume
        lastSpeechTimeRef.current = Date.now() + 1000; // Give 1 second grace period for user to start speaking
        const checkForSilence = () => {
          if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
            const timeSinceLastSpeech = Date.now() - lastSpeechTimeRef.current;
            if (timeSinceLastSpeech > (silenceDuration * 1000)) {
              console.log(`🔇 Backup silence detection: ${timeSinceLastSpeech}ms since last speech (threshold: ${silenceDuration * 1000}ms) - stopping`);
              stopRecording();
              return;
            }
            // Check again in 500ms
            setTimeout(checkForSilence, 500);
          }
        };
        // Start backup silence detection after 2 seconds (to allow user to start speaking after beep)
        setTimeout(checkForSilence, 2000);
      
    } catch (error) {
      console.error('Failed to start recording:', error);
      setIsRecording(false);
      setError('Failed to start recording');
    }
  };

  const stopRecording = () => {
    console.log('🛑 Stop recording called - checking state...');
    console.log('MediaRecorder state:', mediaRecorderRef.current?.state);
    console.log('isRecording state:', isRecording);
    
    if (!mediaRecorderRef.current) {
      console.log('⚠️ Cannot stop recording - no MediaRecorder');
      return;
    }
    
    if (mediaRecorderRef.current.state !== 'recording') {
      console.log('⚠️ Cannot stop recording - MediaRecorder not in recording state:', mediaRecorderRef.current.state);
      return;
    }
    
    console.log('🛑 Stopping recording...');
    setIsRecording(false);
    
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
      console.log('⏰ Cleared silence timer');
    }
    
    if (maxRecordingTimerRef.current) {
      clearTimeout(maxRecordingTimerRef.current);
      maxRecordingTimerRef.current = null;
      console.log('⏰ Cleared max recording timer');
    }
    
    try {
      mediaRecorderRef.current.stop();
      console.log('✅ MediaRecorder stopped successfully');
    } catch (error) {
      console.error('❌ Error stopping MediaRecorder:', error);
    }
  };

  const processRecording = async () => {
    if (audioChunksRef.current.length === 0) return;
    
    setIsProcessing(true);
    
    try {
      // Step 1: Transcribe audio to text
      setIsTranscribing(true);
      const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm;codecs=opus' });
      
      const transcribeFormData = new FormData();
      transcribeFormData.append('audio_file', audioBlob, 'recording.webm');
      
      console.log('🎤 Transcribing audio...');
      
      const transcribeResponse = await fetch(`${API_BASE_URL}${API_ENDPOINTS.TRANSCRIBE_AUDIO}`, {
        method: 'POST',
        body: transcribeFormData,
      });
      
      if (!transcribeResponse.ok) {
        throw new Error('Failed to transcribe audio');
      }
      
      const transcribeData = await transcribeResponse.json();
      
      if (transcribeData.status !== 'success' || !transcribeData.transcript) {
        throw new Error(transcribeData.error || 'No speech detected');
      }
      
      const transcript = transcribeData.transcript;
      setTranscript(transcript);
      setIsTranscribing(false);
      
      console.log('📝 Transcription complete:', transcript);
      
      // Brief pause to show transcription result
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Step 2: Process the transcribed text through the character pipeline
      setIsGeneratingResponse(true);
      console.log('🧠 Processing response...');
      
      const textResponse = await fetch(`${API_BASE_URL}${API_ENDPOINTS.ASK_QUESTION_TEXT}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          character_id: characterId,
          question: transcript
        }),
      });
      
      if (!textResponse.ok) {
        throw new Error('Failed to process question');
      }
      
      const textData = await textResponse.json();
      
      if (textData.status === 'success') {
        setLastResponse(textData.text_response || '');
        setIsGeneratingResponse(false);
        
        // Play audio response if available
        if (textData.audio_base64 && autoPlayEnabled) {
          await playAudioResponse(textData.audio_base64);
        } else {
          // If auto-play is disabled, restart wake word detection immediately
          console.log('🔄 Auto-play disabled - restarting wake word detection');
          isProcessingWakeWordRef.current = false; // Reset processing flag
          if (isListening && !speechRecognitionRef.current) {
            setTimeout(() => {
              startWakeWordDetection();
            }, 500);
          }
        }
      } else {
        throw new Error(textData.error || 'Unknown error');
      }
      
    } catch (err) {
      console.error('Error processing recording:', err);
      setError(err instanceof Error ? err.message : 'Failed to process recording');
      
      // Restart wake word detection after error
      console.log('🔄 Error occurred - restarting wake word detection');
      isProcessingWakeWordRef.current = false; // Reset processing flag
      if (isListening && !speechRecognitionRef.current) {
        setTimeout(() => {
          startWakeWordDetection();
        }, 1000);
      }
    } finally {
      setIsProcessing(false);
      setIsTranscribing(false);
      setIsGeneratingResponse(false);
      audioChunksRef.current = [];
    }
  };

  const startSpeakingWaveform = () => {
    const animate = () => {
      if (!isPlayingResponse) {
        setSpeakingWaveform(Array(12).fill(0));
        return;
      }
      
      // Generate more dramatic waveform data that looks like speech
      const newWaveform = Array(12).fill(0).map((_, index) => {
        // Create more pronounced speech-like patterns
        const base = Math.random() * 0.9 + 0.3; // Base level between 0.3-1.2
        const timeVariation = Math.sin(Date.now() * 0.02 + index * 0.5) * 0.4;
        const randomVariation = Math.sin(Date.now() * 0.015 + Math.random() * 15) * 0.5;
        const result = base + timeVariation + randomVariation;
        return Math.max(0.2, Math.min(1.5, result)); // Allow values up to 1.5 for more dramatic effect
      });
      
      setSpeakingWaveform(newWaveform);
      waveformAnimationRef.current = requestAnimationFrame(animate);
    };
    
    animate();
  };

  const playAudioResponse = async (audioBase64: string): Promise<void> => {
    return new Promise((resolve, reject) => {
      try {
        // Stop any currently playing audio
        if (currentAudioRef.current) {
          currentAudioRef.current.pause();
          currentAudioRef.current.currentTime = 0;
        }
        
        setIsPlayingResponse(true);
        
        // Convert base64 to audio URL
        const audioData = atob(audioBase64.includes(',') ? audioBase64.split(',')[1] : audioBase64);
        const audioArray = new Uint8Array(audioData.length);
        for (let i = 0; i < audioData.length; i++) {
          audioArray[i] = audioData.charCodeAt(i);
        }
        
        const audioBlob = new Blob([audioArray], { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(audioBlob);
        
        const audio = new Audio(audioUrl);
        currentAudioRef.current = audio;
        
        // Start waveform animation when audio starts playing
        audio.onplay = () => {
          startSpeakingWaveform();
        };
        
        audio.onended = () => {
          setIsPlayingResponse(false);
          setSpeakingWaveform(Array(12).fill(0));
          if (waveformAnimationRef.current) {
            cancelAnimationFrame(waveformAnimationRef.current);
          }
          URL.revokeObjectURL(audioUrl);
          
          // Restart wake word detection after audio finishes
          console.log('🔄 Audio finished - restarting wake word detection');
          isProcessingWakeWordRef.current = false; // Reset processing flag
          if (isListening && !speechRecognitionRef.current) {
            setTimeout(() => {
              startWakeWordDetection();
            }, 500);
          }
          
          resolve();
        };
        
        audio.onerror = () => {
          setIsPlayingResponse(false);
          setSpeakingWaveform(Array(12).fill(0));
          if (waveformAnimationRef.current) {
            cancelAnimationFrame(waveformAnimationRef.current);
          }
          URL.revokeObjectURL(audioUrl);
          reject(new Error('Audio playback failed'));
        };
        
        audio.play().catch(reject);
        
      } catch (err) {
        setIsPlayingResponse(false);
        setSpeakingWaveform(Array(12).fill(0));
        reject(err);
      }
    });
  };

  const toggleListening = () => {
    if (isListening) {
      setIsListening(false);
      if (isRecording) {
        stopRecording();
      }
    } else {
      initializeAudioMonitoring();
    }
  };

  const manualRecord = async () => {
    if (isRecording) {
      stopRecording();
    } else if (!isProcessing) {
      await startRecording();
    }
  };

  // Redirect if no character selected
  if (!characterId) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 flex items-center justify-center">
        <div className="text-center text-white">
          <p className="text-xl mb-4">No character selected</p>
          <button
            onClick={() => navigate('/character-selection')}
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg"
          >
            Select Character
          </button>
        </div>
      </div>
    );
  }

  return (
    <>
      <style>{`
        .slider::-webkit-slider-thumb {
          appearance: none;
          height: 20px;
          width: 20px;
          border-radius: 50%;
          background: #3b82f6;
          cursor: pointer;
          border: 2px solid #1e293b;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        .slider::-moz-range-thumb {
          height: 20px;
          width: 20px;
          border-radius: 50%;
          background: #3b82f6;
          cursor: pointer;
          border: 2px solid #1e293b;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
      `}</style>
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white relative overflow-hidden">
      {/* Background Effects */}
      <div className="absolute inset-0 bg-gradient-to-r from-blue-500/5 via-purple-500/5 to-pink-500/5"></div>
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl"></div>
      <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl"></div>

      {/* Header */}
      <div className="relative z-10 flex items-center justify-between p-6">
        <button
          onClick={() => navigate('/character-selection')}
          className="p-3 rounded-full bg-white/5 hover:bg-white/10 transition-all duration-300 backdrop-blur-sm border border-white/10"
        >
          <ArrowLeft size={20} />
        </button>
        
        <div className="text-center">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">
            {characterName}
          </h1>
        </div>

        <button
          onClick={() => setShowSettings(!showSettings)}
          className="p-3 rounded-full bg-white/5 hover:bg-white/10 transition-all duration-300 backdrop-blur-sm border border-white/10"
        >
          <Settings size={20} />
        </button>
      </div>

      {/* Main Interface */}
      <div className="relative z-10 flex flex-col items-center justify-center px-6 py-8">
        {/* Character Avatar with Audio Visualizer */}
        <div className="relative mb-8">
          {/* Circular Waveform Background - Only visible when speaking */}
          {isPlayingResponse && (
            <div className="absolute -inset-8 w-64 h-64">
              <svg className="w-full h-full" viewBox="0 0 300 300">
                {/* Outer rotating background layer */}
                <g className="animate-spin" style={{ animationDuration: '25s' }}>
                  {speakingWaveform.map((amplitude, index) => {
                    const angle = (index / speakingWaveform.length) * 2 * Math.PI;
                    const radius = 80 + amplitude * 60; // Much larger radius range
                    const x1 = 150 + Math.cos(angle) * 90;
                    const y1 = 150 + Math.sin(angle) * 90;
                    const x2 = 150 + Math.cos(angle) * radius;
                    const y2 = 150 + Math.sin(angle) * radius;
                    
                    return (
                      <g key={`bg-${index}`}>
                        {/* Background waveform lines */}
                        <line
                          x1={x1}
                          y1={y1}
                          x2={x2}
                          y2={y2}
                          stroke="url(#waveGradientBg)"
                          strokeWidth="2"
                          opacity={0.6}
                        />
                        {/* Background glow effect */}
                        <line
                          x1={x1}
                          y1={y1}
                          x2={x2}
                          y2={y2}
                          stroke="url(#glowGradient)"
                          strokeWidth="6"
                          opacity={0.3}
                        />
                      </g>
                    );
                  })}
                </g>

                {/* Main dramatic waveform layer */}
                <g>
                  {speakingWaveform.map((amplitude, index) => {
                    const angle = (index / speakingWaveform.length) * 2 * Math.PI;
                    const radius = 85 + amplitude * 70; // Much more dramatic range
                    const x1 = 150 + Math.cos(angle) * 95;
                    const y1 = 150 + Math.sin(angle) * 95;
                    const x2 = 150 + Math.cos(angle) * radius;
                    const y2 = 150 + Math.sin(angle) * radius;
                    
                    return (
                      <g key={index}>
                        {/* Glow effect behind main lines */}
                        <line
                          x1={x1}
                          y1={y1}
                          x2={x2}
                          y2={y2}
                          stroke="url(#mainGlow)"
                          strokeWidth="8"
                          opacity={0.4}
                        />
                        {/* Main waveform lines - much thicker */}
                        <line
                          x1={x1}
                          y1={y1}
                          x2={x2}
                          y2={y2}
                          stroke="url(#waveGradient)"
                          strokeWidth="4"
                          opacity={0.9}
                        />
                        {/* Secondary waveform lines for depth */}
                        <line
                          x1={150 + Math.cos(angle) * 100}
                          y1={150 + Math.sin(angle) * 100}
                          x2={150 + Math.cos(angle) * (radius * 0.85)}
                          y2={150 + Math.sin(angle) * (radius * 0.85)}
                          stroke="url(#waveGradient2)"
                          strokeWidth="2"
                          opacity={0.7}
                        />
                        {/* Large glowing dots at the ends */}
                        <circle
                          cx={x2}
                          cy={y2}
                          r={3 + amplitude * 4}
                          fill="url(#dotGradient)"
                          opacity={0.9}
                        />
                        {/* Inner glow for dots */}
                        <circle
                          cx={x2}
                          cy={y2}
                          r={6 + amplitude * 6}
                          fill="url(#dotGlow)"
                          opacity={0.4}
                        />
                      </g>
                    );
                  })}
                </g>
                
                {/* Counter-rotating layer with more drama */}
                <g className="animate-spin" style={{ animationDuration: '12s', animationDirection: 'reverse' }}>
                  {speakingWaveform.map((amplitude, index) => {
                    if (index % 3 !== 0) return null; // Show every third line
                    const angle = (index / speakingWaveform.length) * 2 * Math.PI;
                    const radius = 70 + amplitude * 50;
                    const x1 = 150 + Math.cos(angle) * 80;
                    const y1 = 150 + Math.sin(angle) * 80;
                    const x2 = 150 + Math.cos(angle) * radius;
                    const y2 = 150 + Math.sin(angle) * radius;
                    
                    return (
                      <g key={`counter-${index}`}>
                        {/* Glow effect */}
                        <line
                          x1={x1}
                          y1={y1}
                          x2={x2}
                          y2={y2}
                          stroke="url(#counterGlow)"
                          strokeWidth="6"
                          opacity={0.3}
                        />
                        {/* Main line */}
                        <line
                          x1={x1}
                          y1={y1}
                          x2={x2}
                          y2={y2}
                          stroke="url(#waveGradient3)"
                          strokeWidth="3"
                          opacity={0.8}
                        />
                      </g>
                    );
                  })}
                </g>

                {/* Additional pulsing layer */}
                <g className="animate-pulse" style={{ animationDuration: '1.5s' }}>
                  {speakingWaveform.map((amplitude, index) => {
                    if (index % 4 !== 0) return null; // Show every fourth line
                    const angle = (index / speakingWaveform.length) * 2 * Math.PI;
                    const radius = 60 + amplitude * 40;
                    const x1 = 150 + Math.cos(angle) * 75;
                    const y1 = 150 + Math.sin(angle) * 75;
                    const x2 = 150 + Math.cos(angle) * radius;
                    const y2 = 150 + Math.sin(angle) * radius;
                    
                    return (
                      <line
                        key={`pulse-${index}`}
                        x1={x1}
                        y1={y1}
                        x2={x2}
                        y2={y2}
                        stroke="url(#pulseGradient)"
                        strokeWidth="2"
                        opacity={0.6}
                      />
                    );
                  })}
                </g>
                
                {/* Gradient definitions */}
                <defs>
                  <linearGradient id="waveGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.8" />
                    <stop offset="50%" stopColor="#8b5cf6" stopOpacity="0.6" />
                    <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.4" />
                  </linearGradient>
                  <linearGradient id="waveGradient2" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.6" />
                    <stop offset="50%" stopColor="#3b82f6" stopOpacity="0.4" />
                    <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0.2" />
                  </linearGradient>
                  <linearGradient id="waveGradient3" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#f59e0b" stopOpacity="0.4" />
                    <stop offset="50%" stopColor="#8b5cf6" stopOpacity="0.3" />
                    <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.2" />
                  </linearGradient>
                  <linearGradient id="waveGradientBg" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#1e293b" stopOpacity="0.4" />
                    <stop offset="50%" stopColor="#3b82f6" stopOpacity="0.2" />
                    <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0.1" />
                  </linearGradient>
                  <radialGradient id="dotGradient" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stopColor="#ffffff" stopOpacity="0.8" />
                    <stop offset="50%" stopColor="#3b82f6" stopOpacity="0.6" />
                    <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0.2" />
                  </radialGradient>
                  <radialGradient id="dotGlow" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stopColor="#ffffff" stopOpacity="0.6" />
                    <stop offset="50%" stopColor="#3b82f6" stopOpacity="0.4" />
                    <stop offset="100%" stopColor="transparent" />
                  </radialGradient>
                  <linearGradient id="mainGlow" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.8" />
                    <stop offset="100%" stopColor="transparent" />
                  </linearGradient>
                  <linearGradient id="glowGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.6" />
                    <stop offset="100%" stopColor="transparent" />
                  </linearGradient>
                  <linearGradient id="counterGlow" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#f59e0b" stopOpacity="0.7" />
                    <stop offset="100%" stopColor="transparent" />
                  </linearGradient>
                  <linearGradient id="pulseGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#ec4899" stopOpacity="0.8" />
                    <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0.6" />
                  </linearGradient>
                </defs>
                
                {/* Circular wave rings */}
                {[1, 2, 3].map((ring, index) => {
                  const ringRadius = 80 + (index * 15) + (Math.max(...speakingWaveform) * 10);
                  return (
                    <circle
                      key={ring}
                      cx="100"
                      cy="100"
                      r={ringRadius}
                      fill="none"
                      stroke={`hsl(${220 + index * 20}, 70%, 60%)`}
                      strokeWidth="1"
                      opacity={0.3 - index * 0.1}
                      className="animate-pulse"
                      style={{
                        animationDelay: `${index * 0.2}s`,
                        animationDuration: '2s'
                      }}
                    />
                  );
                })}
              </svg>
            </div>
          )}

          {/* Character Image */}
          <div className="relative w-48 h-48 rounded-full overflow-hidden border-4 border-white/20 shadow-2xl z-10">
            {characterImage ? (
              <img 
                src={characterImage} 
                alt={characterName}
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="w-full h-full bg-gradient-to-br from-slate-700 to-slate-800 flex items-center justify-center">
                <span className="text-4xl font-bold text-white/50">{characterName.charAt(0)}</span>
              </div>
            )}
            
            {/* Audio level overlay */}
            <div 
              className="absolute inset-0 bg-gradient-to-r from-blue-500/30 to-purple-500/30 rounded-full transition-opacity duration-200"
              style={{ opacity: audioLevel > 0.01 ? audioLevel * 2 : 0 }}
            />
          </div>

          {/* Pulsing rings based on audio level and state */}
          <div 
            className="absolute inset-0 rounded-full border-2 border-blue-400/50 transition-all duration-300 z-20"
            style={{
              transform: `scale(${1 + audioLevel * 0.5})`,
              opacity: (audioLevel > 0.01 || isRecording) ? 0.8 : 0.3
            }}
          />
          <div 
            className="absolute -inset-2 rounded-full border border-purple-400/30 transition-all duration-500 z-20"
            style={{
              transform: `scale(${1 + audioLevel * 0.3})`,
              opacity: (audioLevel > 0.02 || isRecording) ? 0.6 : 0.2
            }}
          />

          {/* Status indicator overlay */}
          <div className="absolute -bottom-2 -right-2">
            {isTranscribing ? (
              <div className="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center shadow-lg border-2 border-white/20">
                <Loader className="w-6 h-6 animate-spin text-white" />
              </div>
            ) : isGeneratingResponse ? (
              <div className="w-12 h-12 bg-yellow-500 rounded-full flex items-center justify-center animate-pulse shadow-lg border-2 border-white/20">
                <Zap className="w-6 h-6 text-white" />
              </div>
            ) : isRecording ? (
              <div className="w-12 h-12 bg-red-500 rounded-full flex items-center justify-center animate-pulse shadow-lg border-2 border-white/20">
                <Mic className="w-6 h-6 text-white" />
              </div>
            ) : isPlayingResponse ? (
              <div className="w-12 h-12 bg-green-500 rounded-full flex items-center justify-center animate-pulse shadow-lg border-2 border-white/20">
                <Volume2 className="w-6 h-6 text-white" />
              </div>
            ) : (
              <div className={`w-12 h-12 ${isListening ? 'bg-blue-500' : 'bg-slate-600'} rounded-full flex items-center justify-center shadow-lg border-2 border-white/20`}>
                {isListening ? <Mic className="w-6 h-6 text-white" /> : <MicOff className="w-6 h-6 text-white" />}
              </div>
            )}
          </div>
        </div>

        {/* Debug Info */}
        {(isRecording || isListening) && (
          <div className="text-center mb-4 text-sm text-slate-400">
            <div>Volume: {audioLevel.toFixed(4)} | Threshold: {volumeThreshold}</div>
            <div>Voice Detected: {audioLevel > volumeThreshold ? '✅' : '❌'}</div>
            {!isRecording && (
              <div className="mt-2">
                <span className="text-xs">Microphone test: {audioLevel > 0.001 ? '🎤 Working' : '❌ No input'}</span>
              </div>
            )}
          </div>
        )}

        {/* Status */}
        <div className="text-center mb-8">
          {isTranscribing ? (
            <div className="flex flex-col items-center gap-3">
              <span className="text-xl font-medium">Listening...</span>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
                <div className="w-6 h-0.5 bg-gradient-to-r from-blue-400 to-transparent"></div>
                <div className="w-2 h-2 bg-slate-600 rounded-full"></div>
              </div>
            </div>
          ) : isGeneratingResponse ? (
            <div className="flex flex-col items-center gap-3">
              <span className="text-xl font-medium">Thinking...</span>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                <div className="w-6 h-0.5 bg-gradient-to-r from-blue-400 to-yellow-400"></div>
                <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse"></div>
              </div>
            </div>
          ) : isRecording ? (
            <div className="flex flex-col items-center gap-2">
              <span className="text-xl font-medium text-red-300">🎤 Recording - Speak Now</span>
              <div className="flex gap-1">
                <div className="w-1 h-4 bg-red-500 rounded-full animate-pulse"></div>
                <div className="w-1 h-6 bg-red-500 rounded-full animate-pulse" style={{animationDelay: '0.1s'}}></div>
                <div className="w-1 h-5 bg-red-500 rounded-full animate-pulse" style={{animationDelay: '0.2s'}}></div>
              </div>
              <div className="text-sm text-slate-400">
                Will stop after {silenceDuration} seconds of silence
              </div>
            </div>
          ) : isPlayingResponse ? (
            <div className="flex flex-col items-center gap-2">
              <span className="text-xl font-medium">Speaking...</span>
              <div className="flex gap-1">
                <div className="w-1 h-4 bg-green-500 rounded-full animate-pulse"></div>
                <div className="w-1 h-6 bg-green-500 rounded-full animate-pulse" style={{animationDelay: '0.1s'}}></div>
                <div className="w-1 h-5 bg-green-500 rounded-full animate-pulse" style={{animationDelay: '0.2s'}}></div>
              </div>
            </div>
          ) : isListening ? (
            <div className="flex flex-col items-center gap-2">
              <span className="text-lg text-slate-300">Say "Hey Oppenheimer"</span>
              {isWakeWordListening && (
                <div className="flex items-center gap-2 text-sm text-blue-400">
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
                  <span>Listening for wake word...</span>
                </div>
              )}
            </div>
          ) : (
            <span className="text-lg text-slate-500">Voice detection off</span>
          )}
        </div>

        {/* Controls */}
        <div className="flex gap-3 mb-8">
          <button
            onClick={toggleListening}
            className={`px-8 py-3 rounded-full font-medium transition-all duration-300 backdrop-blur-sm border ${
              isListening 
                ? 'bg-red-500/20 hover:bg-red-500/30 text-red-300 border-red-500/30' 
                : 'bg-green-500/20 hover:bg-green-500/30 text-green-300 border-green-500/30'
            }`}
          >
            {isListening ? 'Stop Wake Word' : 'Start Wake Word'}
          </button>
          
          <button
            onClick={manualRecord}
            disabled={!isListening || isTranscribing || isGeneratingResponse}
            className={`px-8 py-3 rounded-full font-medium transition-all duration-300 backdrop-blur-sm border ${
              isRecording
                ? 'bg-red-500/20 hover:bg-red-500/30 text-red-300 border-red-500/30'
                : 'bg-blue-500/20 hover:bg-blue-500/30 text-blue-300 border-blue-500/30 disabled:bg-slate-500/10 disabled:text-slate-500 disabled:border-slate-500/20 disabled:cursor-not-allowed'
            }`}
          >
            {isRecording ? 'Force Stop' : 'Push to Talk'}
          </button>
          
          <button
            onClick={() => setAutoPlayEnabled(!autoPlayEnabled)}
            className={`p-3 rounded-full font-medium transition-all duration-300 backdrop-blur-sm border ${
              autoPlayEnabled
                ? 'bg-purple-500/20 hover:bg-purple-500/30 text-purple-300 border-purple-500/30'
                : 'bg-slate-500/20 hover:bg-slate-500/30 text-slate-300 border-slate-500/30'
            }`}
          >
            {autoPlayEnabled ? <Volume2 size={20} /> : <VolumeX size={20} />}
          </button>
        </div>

        {/* Conversation Display */}
        {(transcript || lastResponse) && (
          <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-6 max-w-2xl w-full border border-white/10 shadow-xl">
            {transcript && (
              <div className="mb-6">
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 rounded-full bg-blue-500/20 flex items-center justify-center">
                    <span className="text-sm font-medium">You</span>
                  </div>
                  <div className="flex-1">
                    <p className="text-white/90 leading-relaxed">{transcript}</p>
                  </div>
                </div>
              </div>
            )}
            
            {lastResponse && (
              <div>
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 rounded-full bg-purple-500/20 flex items-center justify-center overflow-hidden">
                    {characterImage ? (
                      <img src={characterImage} alt={characterName} className="w-full h-full object-cover" />
                    ) : (
                      <span className="text-sm font-medium">{characterName.charAt(0)}</span>
                    )}
                  </div>
                  <div className="flex-1">
                    <p className="text-white/90 leading-relaxed">{lastResponse}</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="bg-red-500/10 backdrop-blur-sm border border-red-500/30 rounded-xl p-4 max-w-md w-full mt-4">
            <p className="text-red-300">{error}</p>
            <button
              onClick={() => setError(null)}
              className="mt-2 text-red-400 hover:text-red-300 text-sm underline"
            >
              Dismiss
            </button>
          </div>
        )}
      </div>

      {/* Settings Modal */}
      {showSettings && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
          <div className="bg-slate-800/90 backdrop-blur-xl rounded-2xl p-6 max-w-md w-full border border-white/10 shadow-2xl">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold">Voice Settings</h3>
              <button
                onClick={() => setShowSettings(false)}
                className="p-2 rounded-full hover:bg-white/10 transition-colors"
              >
                <X size={20} />
              </button>
            </div>
            
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium mb-3 text-slate-300">
                  Voice Sensitivity: {volumeThreshold.toFixed(3)}
                </label>
                <input
                  type="range"
                  min="0.01"
                  max="0.2"
                  step="0.005"
                  value={volumeThreshold}
                  onChange={(e) => setVolumeThreshold(parseFloat(e.target.value))}
                  className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer slider"
                />
                <div className="flex justify-between text-xs text-slate-400 mt-1">
                  <span>Less sensitive</span>
                  <span>More sensitive</span>
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-3 text-slate-300">
                  Silence Duration: {silenceDuration}s
                </label>
                <input
                  type="range"
                  min="0.5"
                  max="5"
                  step="0.1"
                  value={silenceDuration}
                  onChange={(e) => setSilenceDuration(parseFloat(e.target.value))}
                  className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer slider"
                />
                <div className="flex justify-between text-xs text-slate-400 mt-1">
                  <span>0.5s</span>
                  <span>5s</span>
                </div>
              </div>

              <div className="flex items-center justify-between py-2">
                <span className="text-sm font-medium text-slate-300">Auto-play responses</span>
                <button
                  onClick={() => setAutoPlayEnabled(!autoPlayEnabled)}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    autoPlayEnabled ? 'bg-blue-500' : 'bg-slate-600'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      autoPlayEnabled ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>
            </div>
          </div>
                 </div>
       )}
     </div>
     </>
   );
};

export default VoiceInteraction; 