import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import { PredictionProvider } from "./store/prediction";
import Overview from "./pages/Overview";
import Assessment from "./pages/Assessment";
import Result from "./pages/Result";
import Explainability from "./pages/Explainability";
import Performance from "./pages/Performance";
import DatasetQuality from "./pages/DatasetQuality";
import ModelInfo from "./pages/ModelInfo";
import ResponsibleUse from "./pages/ResponsibleUse";

export default function App() {
  return (
    <PredictionProvider>
      <Layout>
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/assessment" element={<Assessment />} />
          <Route path="/result" element={<Result />} />
          <Route path="/explainability" element={<Explainability />} />
          <Route path="/performance" element={<Performance />} />
          <Route path="/dataset" element={<DatasetQuality />} />
          <Route path="/model" element={<ModelInfo />} />
          <Route path="/responsible-use" element={<ResponsibleUse />} />
        </Routes>
      </Layout>
    </PredictionProvider>
  );
}
