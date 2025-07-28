import React from "react";
import PropTypes from "prop-types";
import { motion } from "framer-motion";
import { ArrowRight, Github, Globe } from "lucide-react";
import { Link } from "react-router-dom"; // ✅ for routing

const projects = [
  {
    title: "🚀 AI-Powered Disaster Assessment and Cost Estimation System",
    description:
      "Instance Segmentation For Building Damage Assessment and Cost Estimation Using Deep Learning and Computer Vision.",
    link: "https://i.imgur.com/IfohMTo.jpeg",
    color: "#5196fd",
    githubLink: "https://github.com/AI-Studio-DeployForce/training-pipeline",
    liveLink: "#",
    id: 1,
  },
  {
    title: "VisionAid-VQA",
    description:
      "Inclusive Visual Question Answering Using Deep Learning and Multimodal Attention Mechanisms.",
    link: "https://i.imgur.com/vHqlHra.jpeg",
    color: "#ed649e",
    githubLink: "https://github.com/zagarsuren/visionaid-vqa",
    liveLink: "#",
    id: 2,
  },
  {
    title: "Chest Xray Classification",
    description:
      "A Streamlit-powered web application that classifies chest X-ray images using state-of-the-art deep learning models. The system supports multiple backbone architectures and ensemble predictions for robust diagnosis across five major thoracic conditions.",
    link: "https://i.imgur.com/guH3gk8.jpeg",
    color: "#ed649e",
    githubLink: "https://github.com/zagarsuren/chest-xray-app",
    liveLink: "https://www.youtube.com/watch?v=e93Gh2Yy7sc",
    id: 3,
  },
  {
    title: "Trader AI: Reinforcement Learning for Stock Trading",
    description:
      "Reinforcement learning-based stock trading agent that learns optimal trading strategies through simulation. The agent uses deep Q-learning to maximize returns by analyzing historical stock data and making buy/sell decisions.",
    link: "https://i.imgur.com/zWnE4KN.jpeg",
    color: "#8f89ff",
    githubLink: "https://github.com/zagarsuren/trader_ai_app_dqnq",
    liveLink: "https://www.youtube.com/watch?v=7R0psq1VRms",
    id: 4,
  },
  {
    title: "Satellite Image Analysis and Object Detection",
    description:
      "A comprehensive project that combines satellite imagery analysis with object detection techniques.",
    link: "https://i.imgur.com/XVS8aHc.jpeg",
    color: "#fff",
    githubLink: "#",
    liveLink: "#",
    id: 5,
  },
  {
    title: "HerdWatch - UTS AI Showcase 2024",
    description:
      "This project aims to develop a system for livestock counting and behaviour detection using computer vision algorithms. The proposed system utilises state-of-the-art deep learning techniques to process images or video footage captured from surveillance cameras installed in livestock facilities or drones.",
    link: "https://i.imgur.com/UTshSsS.jpeg",
    color: "#ed649e",
    githubLink: "https://github.com/zagarsuren/uts-ai-showcase-herdwatch",
    liveLink: "https://www.youtube.com/watch?v=E_jsOMbI-yE",
    id: 6,
  },  
];

export default function Projects() {
  return (
    <section className="bg-black text-white py-24 px-4 lg:px-8 font-mono">
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
        <h2 className="text-2xl font-semibold font-mono">{title}</h2>
        <p className="text-gray-400 text-sm font-mono">{description}</p>

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
            Demo
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
