import { BrowserRouter as Router, Routes, Route } from 'react-router';
import Home from './pages/Home';
import UrlScanTab from './pages/UrlScanTab';
import Navbar from './components/Navbar';
import UrlListScanTab from './pages/UrlListScanTab';

function App() {
  return (
    <Router>
      <Navbar/>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/scan/url" element={<UrlScanTab />} />
        <Route path="/scan/urllist" element={<UrlListScanTab />} />
      </Routes>
    </Router>
  );
}

export default App;