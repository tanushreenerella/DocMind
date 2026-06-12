"use client";

import { useState, useRef, useCallback } from "react";
import { Mic, Loader2 } from "lucide-react";
import { transcribeAudio } from "@/lib/api";

interface VoiceInputProps {
  onSend: (transcript: string) => void;
  disabled?: boolean;
}

const SILENCE_THRESHOLD = 15;
const SILENCE_DURATION_MS = 1500;
const WARMUP_MS = 1000;

export default function VoiceInput({ onSend, disabled = false }: VoiceInputProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const stoppedRef = useRef(false);

  const cleanup = useCallback(() => {
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
    }
  }, []);

  const stopAndSend = useCallback(() => {
    if (stoppedRef.current) return;
    stoppedRef.current = true;
    cleanup();
    const rec = mediaRecorderRef.current;
    if (rec && rec.state !== "inactive") rec.stop();
  }, [cleanup]);

  const startSilenceDetection = useCallback(
    (stream: MediaStream) => {
      const audioCtx = new AudioContext();
      audioCtxRef.current = audioCtx;
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      audioCtx.createMediaStreamSource(stream).connect(analyser);

      const data = new Uint8Array(analyser.frequencyBinCount);
      let silenceStart: number | null = null;

      const check = () => {
        if (stoppedRef.current) return;
        analyser.getByteFrequencyData(data);
        const avg = data.reduce((a, b) => a + b, 0) / data.length;

        if (avg < SILENCE_THRESHOLD) {
          if (!silenceStart) silenceStart = Date.now();
          else if (Date.now() - silenceStart > SILENCE_DURATION_MS) {
            stopAndSend();
            return;
          }
        } else {
          silenceStart = null;
        }
        animFrameRef.current = requestAnimationFrame(check);
      };

      setTimeout(() => {
        if (!stoppedRef.current) animFrameRef.current = requestAnimationFrame(check);
      }, WARMUP_MS);
    },
    [stopAndSend]
  );

  const blobToBase64 = (blob: Blob): Promise<string> =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve((reader.result as string).split(",")[1]);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });

  const startRecording = async () => {
    if (disabled || isProcessing) return;
    stoppedRef.current = false;
    chunksRef.current = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        setIsRecording(false);
        setIsProcessing(true);
        try {
          const blob = new Blob(chunksRef.current, { type: "audio/webm" });
          const base64 = await blobToBase64(blob);
          const { transcript } = await transcribeAudio(base64);
          if (transcript?.trim()) onSend(transcript.trim());
        } catch {
          // silently swallow transcription errors
        } finally {
          setIsProcessing(false);
        }
      };

      recorder.start();
      setIsRecording(true);
      startSilenceDetection(stream);
    } catch {
      // mic access denied — do nothing
    }
  };

  return (
    <button
      onClick={isRecording ? stopAndSend : startRecording}
      disabled={disabled || isProcessing}
      title={
        isProcessing
          ? "Sending…"
          : isRecording
          ? "Listening… pause to auto-send"
          : "Click to speak"
      }
      className={`
        p-2 rounded-full transition-all duration-200
        ${
          isRecording
            ? "text-red-500 animate-pulse bg-red-50"
            : isProcessing
            ? "text-gray-400 cursor-not-allowed"
            : "text-gray-400 hover:text-indigo-500 hover:bg-indigo-50"
        }
        disabled:opacity-50
      `}
    >
      {isProcessing ? (
        <Loader2 className="w-5 h-5 animate-spin" />
      ) : (
        <Mic className="w-5 h-5" />
      )}
    </button>
  );
}
