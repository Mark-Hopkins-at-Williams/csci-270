import Csci270 from "./components/Csci270";
import { createHashRouter, RouterProvider } from "react-router-dom";
import "./styles.css";

const App = () => {
  const router = createHashRouter([
    {
      path: "/",
      element: <Csci270 />,
    },
  ]);

  return <RouterProvider router={router} />;
};

export default App;
