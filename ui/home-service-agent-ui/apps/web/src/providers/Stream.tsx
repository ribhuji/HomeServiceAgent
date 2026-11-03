"use client";

import React, { ReactNode, createContext, useContext } from "react";
import { useQueryState } from "nuqs";
import { Message } from "@langchain/langgraph-sdk";
import { useStream } from "@langchain/langgraph-sdk/react";

export type StateType = { messages: Message[] };
const useTypedStream = useStream<StateType>;
type StreamContextType = ReturnType<typeof useTypedStream>;
const StreamContext = createContext<StreamContextType | undefined>(undefined);

const DEFAULT_API_URL = "http://localhost:2024";
const DEFAULT_ASSISTANT_ID = "home_service_agent";

export function StreamProvider({ children }: { children: ReactNode }) {
  const [threadId, setThreadId] = useQueryState("threadId");
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || DEFAULT_API_URL;
  const assistantId = process.env.NEXT_PUBLIC_ASSISTANT_ID || DEFAULT_ASSISTANT_ID;
  const stream = useTypedStream({
    apiUrl,
    assistantId,
    threadId: threadId ?? null,
    onThreadId: setThreadId,
  });

  return <StreamContext.Provider value={stream}>{children}</StreamContext.Provider>;
}

export function useStreamContext(): StreamContextType {
  const context = useContext(StreamContext);
  if (!context) throw new Error("useStreamContext must be used within StreamProvider");
  return context;
}
