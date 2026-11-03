"use client";

import { FormEvent, useState } from "react";
import { Message } from "@langchain/langgraph-sdk";
import { useQueryState } from "nuqs";
import { useStreamContext } from "@/providers/Stream";
import { getContentString } from "./utils";

export function Thread() {
  const stream = useStreamContext();
  const [, setThreadId] = useQueryState("threadId");
  const [input, setInput] = useState("");

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (!input.trim() || stream.isLoading) return;
    const message: Message = { type: "human", content: input.trim() };
    stream.submit(
      { messages: [message] },
      {
        streamMode: ["values"],
        optimisticValues: (previous) => ({
          ...previous,
          messages: [...(previous.messages ?? []), message],
        }),
      },
    );
    setInput("");
  };

  return (
    <main className="flex h-screen flex-col bg-background">
      <header className="border-b px-6 py-4">
        <button className="text-xl font-semibold" onClick={() => setThreadId(null)}>
          Home Service Agent
        </button>
      </header>
      <section className="flex-1 overflow-y-auto px-6 py-8">
        <div className="mx-auto flex max-w-3xl flex-col gap-4">
          {stream.messages.map((message, index) => {
            const text = getContentString(message.content);
            const mine = message.type === "human";
            return (
              <div
                key={message.id ?? `${message.type}-${index}`}
                className={mine ? "ml-auto max-w-[80%] rounded-2xl bg-muted px-4 py-3" : "max-w-[80%] rounded-2xl border px-4 py-3"}
              >
                {String(text)}
              </div>
            );
          })}
          {stream.isLoading && <div className="rounded-2xl border px-4 py-3 text-muted-foreground">Thinking…</div>}
          {stream.error ? <div className="rounded-2xl border border-red-300 px-4 py-3 text-red-700">{String(stream.error)}</div> : null}
        </div>
      </section>
      <form onSubmit={submit} className="border-t p-4">
        <div className="mx-auto flex max-w-3xl gap-2">
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
            placeholder="Describe the home service you need…"
            className="min-h-12 flex-1 resize-none rounded-xl border bg-background p-3 outline-none focus:ring-2"
            disabled={stream.isLoading}
          />
          <button type="submit" disabled={stream.isLoading || !input.trim()} className="self-end rounded-xl bg-primary px-5 py-3 text-primary-foreground disabled:opacity-50">
            Send
          </button>
        </div>
      </form>
    </main>
  );
}
