import { Link } from "react-router";

const Navbar = () => {
  return (
    <header className="w-full py-6 px-8 flex justify-between items-center bg-gradient-to-br from-gray-900 via-gray-800 to-blue-900 shadow">
      <span className="text-2xl font-extrabold text-blue-400 tracking-tight">
        <Link to="/" className="hover:underline">
          Malicious<span className="text-white">Detection</span>
        </Link>
      </span>
      <nav className="space-x-4">
        <Link to="/scan/url" className={navBtnStyle}>URL Scan</Link>
        <Link to="/scan/urllist" className={navBtnStyle}>URL List Scan</Link>

      </nav>
    </header>
  );
};

const navBtnStyle =
  "inline-block px-4 py-2 rounded-full bg-blue-500 text-white font-semibold hover:bg-blue-600 transition";

export default Navbar;