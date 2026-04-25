"use client";

import { create } from "zustand";
import type { Citation } from "@/lib/api";

export type Stage = "input_safety" | "retrieve" | "generate" | "output_safety" | "done";

export interface StageState {
  input_safety?: "pending" | "passed" | "blocked";
  retrieve?: "pending" | "done";
  generate?: "pending" | "done";
  output_safety?: "pending" | "passed" | "blocked";
}

export interface QAPair {
  id: string;
  question: string;
  answer: string;
  citations: Citation[];
  stages: StageState;
  blockedBy: string | null;
  refusal: string | null;
  error: string | null;
  streaming: boolean;
}

interface ChatStore {
  pairs: QAPair[];
  addPair: (question: string) => string;
  updatePair: (id: string, patch: Partial<QAPair>) => void;
  appendToken: (id: string, text: string) => void;
  setStage: (id: string, stage: keyof StageState, value: NonNullable<StageState[keyof StageState]>) => void;
  clear: () => void;
}

export const useChatStore = create<ChatStore>((set) => ({
  pairs: [],
  addPair: (question) => {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    set((s) => ({
      pairs: [
        ...s.pairs,
        {
          id,
          question,
          answer: "",
          citations: [],
          stages: { input_safety: "pending" },
          blockedBy: null,
          refusal: null,
          error: null,
          streaming: true,
        },
      ],
    }));
    return id;
  },
  updatePair: (id, patch) =>
    set((s) => ({
      pairs: s.pairs.map((p) => (p.id === id ? { ...p, ...patch } : p)),
    })),
  appendToken: (id, text) =>
    set((s) => ({
      pairs: s.pairs.map((p) =>
        p.id === id ? { ...p, answer: p.answer + text } : p
      ),
    })),
  setStage: (id, stage, value) =>
    set((s) => ({
      pairs: s.pairs.map((p) =>
        p.id === id ? { ...p, stages: { ...p.stages, [stage]: value } } : p
      ),
    })),
  clear: () => set({ pairs: [] }),
}));
