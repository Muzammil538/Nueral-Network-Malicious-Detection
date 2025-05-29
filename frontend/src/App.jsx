import { BrowserRouter as Router, Routes, Route } from 'react-router';
import Home from './pages/Home';
import UrlScanTab from './pages/UrlScanTab';
import Navbar from './components/Navbar';

function App() {
  return (
    <Router>
      <Navbar/>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/scan/url" element={<UrlScanTab />} />
      </Routes>
    </Router>
  );
}

export default App;