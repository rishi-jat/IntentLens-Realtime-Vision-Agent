/**
 * IntentLens — Root application component (v4 — Embodied Agent).
 */

import "./App.css";
import { VideoFeed } from "./components";

function App() {
  return (
    <div className="app-root">
      <main className="app-main">
        <VideoFeed />
      </main>
    </div>
  );
}

export default App;
