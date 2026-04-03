/**
 * App.jsx — Main application component with routing
 */
import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navigation from "./components/Navigation";
import Dashboard from "./components/Dashboard";
import Guide from "./components/Guide";

export default function App() {
  return (
    <BrowserRouter>
      <div style={{
        display: "flex",
        flexDirection: "column",
        height: "100vh",
        background: "#f9fafb",
        fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
      }}>
        {/* Navigation Bar */}
        <Navigation />

        {/* Routes */}
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/guide" element={<Guide />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}
