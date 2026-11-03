"use client";

import { Thread } from "@/components/thread";
import { StreamProvider } from "@/providers/Stream";
import React from "react";

export default function DemoPage(): React.ReactNode {
  return (
    <React.Suspense fallback={<div>Loading (layout)...</div>}>
      <StreamProvider>
        <Thread />
      </StreamProvider>
    </React.Suspense>
  );
}
