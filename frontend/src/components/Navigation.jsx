/**
 * Navigation.jsx — Navigation bar for vLLM Auto-Tuner Dashboard
 */
import React from "react";
import { Link, useLocation } from "react-router-dom";

export default function Navigation() {
  const location = useLocation();

  const isActive = (path) => {
    return location.pathname === path ? "active" : "";
  };

  const navStyles = {
    container: {
      background: "#fff",
      borderBottom: "1px solid #e5e7eb",
      padding: "0 20px",
      height: 48,
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      flexShrink: 0,
    },
    leftGroup: {
      display: "flex",
      alignItems: "center",
      gap: 8,
    },
    logo: {
      fontFamily: "monospace",
      fontSize: 11,
      fontWeight: 700,
      color: "#16a34a",
      background: "#dcfce7",
      padding: "3px 9px",
      borderRadius: 2,
      letterSpacing: "0.1em",
      marginRight: 12,
    },
    navLinks: {
      display: "flex",
      gap: 0,
      alignItems: "center",
    },
    navLink: (isActivePath) => ({
      padding: "8px 16px",
      fontSize: 13,
      fontWeight: 500,
      color: isActivePath ? "#16a34a" : "#6b7280",
      textDecoration: "none",
      borderBottom: isActivePath ? "2px solid #16a34a" : "2px solid transparent",
      transition: "all 0.2s",
      cursor: "pointer",
      height: "100%",
      display: "flex",
      alignItems: "center",
    }),
  };

  return (
    <div style={navStyles.container}>
      <div style={navStyles.leftGroup}>
        <div style={navStyles.logo}>vLLM AUTO-TUNER</div>
        <nav style={navStyles.navLinks}>
          <Link
            to="/"
            style={navStyles.navLink(isActive("/") === "active")}
            onMouseEnter={(e) => {
              if (isActive("/") !== "active") {
                e.target.style.color = "#1f2937";
              }
            }}
            onMouseLeave={(e) => {
              if (isActive("/") !== "active") {
                e.target.style.color = "#6b7280";
              }
            }}
          >
            Dashboard
          </Link>
          <Link
            to="/guide"
            style={navStyles.navLink(isActive("/guide") === "active")}
            onMouseEnter={(e) => {
              if (isActive("/guide") !== "active") {
                e.target.style.color = "#1f2937";
              }
            }}
            onMouseLeave={(e) => {
              if (isActive("/guide") !== "active") {
                e.target.style.color = "#6b7280";
              }
            }}
          >
            Guide
          </Link>
          <a
            href="/vllm_sampling_params.html"
            target="_blank"
            rel="noopener noreferrer"
            style={navStyles.navLink(false)}
            onMouseEnter={(e) => {
              e.target.style.color = "#1f2937";
            }}
            onMouseLeave={(e) => {
              e.target.style.color = "#6b7280";
            }}
          >
            Sampling Params
          </a>
        </nav>
      </div>
    </div>
  );
}
