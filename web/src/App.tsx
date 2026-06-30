import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AuthProvider } from "@/auth/AuthContext";
import { RegionProvider } from "@/lib/RegionContext";
import { ProtectedRoute } from "@/auth/ProtectedRoute";
import Home from "./pages/Home";
import BuildHistory from "./pages/BuildHistory";
import Results from "./pages/Results";
import Login from "./pages/Login";
import Register from "./pages/Register";
import Profile from "./pages/Profile";
import Friends from "./pages/Friends";
import Blend from "./pages/Blend";

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <RegionProvider>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/build" element={<BuildHistory />} />
            <Route path="/results" element={<Results />} />
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            <Route
              path="/profile"
              element={
                <ProtectedRoute>
                  <Profile />
                </ProtectedRoute>
              }
            />
            <Route
              path="/friends"
              element={
                <ProtectedRoute>
                  <Friends />
                </ProtectedRoute>
              }
            />
            <Route
              path="/blend/:friendId"
              element={
                <ProtectedRoute>
                  <Blend />
                </ProtectedRoute>
              }
            />
          </Routes>
        </RegionProvider>
      </AuthProvider>
    </BrowserRouter>
  );
}
