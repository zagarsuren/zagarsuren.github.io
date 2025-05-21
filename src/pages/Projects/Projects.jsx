import React from "react";
import PropTypes from "prop-types";
import { motion } from "framer-motion";
import { ArrowRight, Github, Globe } from "lucide-react";
import { Link } from "react-router-dom"; // ✅ for routing

const projects = [
  {
    title: "🚀 Image Segmentation",
    description:
      "A lightweight JavaScript library for creating beautiful, responsive UI components.",
    link: "#",
    color: "#5196fd",
    githubLink: "#",
    liveLink: "#",
    id: 1,
  },
  {
    title: "VisionAid-VQA",
    description:
      "Inclusive Visual Question Answering Using Deep Learning and Multimodal Attention Mechanisms.",
    link: "https://github.com/zagarsuren/visionaid-vqa/blob/main/assets/demo/app.jpeg?raw=true",
    color: "#ed649e",
    githubLink: "https://github.com/zagarsuren/visionaid-vqa",
    liveLink: "#",
    id: 2,
  },
  {
    title: "Chest Xray Classification",
    description:
      "A Streamlit-powered web application that classifies chest X-ray images using state-of-the-art deep learning models. The system supports multiple backbone architectures and ensemble predictions for robust diagnosis across five major thoracic conditions.",
    link: "https://github.com/zagarsuren/chest-xray-app/blob/main/assets/ss1.jpeg?raw=true",
    color: "#ed649e",
    githubLink: "https://github.com/zagarsuren/chest-xray-app",
    liveLink: "#",
    id: 3,
  },
  {
    title: "Trader AI: Reinforcement Learning for Stock Trading",
    description:
      "A sleek portfolio built with React and Tailwind CSS to showcase your skills, projects, and experience in a modern design.",
    link: "https://github.com/zagarsuren/trader_ai_app_dqnq/blob/main/assets/app1.jpg?raw=true",
    color: "#8f89ff",
    githubLink: "#",
    liveLink: "#",
    id: 4,
  },
  {
    title: "Satellite Image Analysis and Object Detection",
    description:
      "A powerful online code editor built with React and Tailwind CSS. Featuring real-time code execution, syntax highlighting, multi-language support, and a sleek UI.",
    link: "#",
    color: "#fff",
    githubLink: "#",
    liveLink: "#",
    id: 5,
  },
  {
    title: "HerdWatch - UTS AI Showcase 2024",
    description:
      "This project aims to develop a system for livestock counting and behaviour detection using computer vision algorithms. The proposed system utilises state-of-the-art deep learning techniques to process images or video footage captured from surveillance cameras installed in livestock facilities or drones.",
    link: "#",
    color: "#ed649e",
    githubLink: "https://github.com/zagarsuren/uts-ai-showcase-herdwatch",
    liveLink: "https://www.youtube.com/watch?v=E_jsOMbI-yE",
    id: 6,
  },  
];

export default function Projects() {
  return (
    <section className="bg-black text-white py-24 px-4 lg:px-8">
      <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-12">
        {projects.map((project, i) => (
          <Card key={`project_${i}`} {...project} index={i + 1} />
        ))}
      </div>
    </section>
  );
}

function Card({ title, description, link, color, githubLink, liveLink, index, id }) {
  return (
    <div className="rounded-xl overflow-hidden bg-zinc-900 shadow-md mt-6 mb-6">
      <div className="relative h-60 md:h-72 overflow-hidden">
        <motion.img
          src={link}
          alt={title}
          className="w-full h-full object-cover"
          initial={{ scale: 1 }}
          whileHover={{ scale: 1.05 }}
          transition={{ duration: 0.4 }}
        />
        <motion.div
          className="absolute inset-0"
          style={{ backgroundColor: color, mixBlendMode: "overlay" }}
          initial={{ opacity: 0 }}
          whileHover={{ opacity: 0.3 }}
          transition={{ duration: 0.3 }}
        />
        <div className="absolute top-4 left-4 bg-black/60 px-3 py-1 rounded-full text-xs font-medium">
          Project {index}
        </div>
      </div>

      <div className="p-6 space-y-4">
        <h2 className="text-2xl font-semibold">{title}</h2>
        <p className="text-gray-400 text-sm">{description}</p>

        <div className="flex gap-4 pt-2">
          <a
            href={githubLink}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 text-blue-400 hover:text-blue-200 text-sm transition"
          >
            <Github size={16} />
            View Code
          </a>

          <a
            href={liveLink}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 text-green-400 hover:text-green-200 text-sm transition"
          >
            <Globe size={16} />
            Live Demo
          </a>

          <Link
            to={`/project/${id}`}
            className="flex items-center gap-1 text-yellow-400 hover:text-yellow-200 text-sm transition ml-auto"
          >
            Read More
            <ArrowRight size={16} />
          </Link>
        </div>
      </div>
    </div>
  );
}

Card.propTypes = {
  title: PropTypes.string.isRequired,
  description: PropTypes.string.isRequired,
  link: PropTypes.string.isRequired,
  color: PropTypes.string.isRequired,
  githubLink: PropTypes.string.isRequired,
  liveLink: PropTypes.string.isRequired,
  index: PropTypes.number.isRequired,
  id: PropTypes.number.isRequired,
};
