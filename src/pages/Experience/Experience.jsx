import React from "react";
import {
  Binary,
  BrainCircuit,
  Cpu,
  Network,
  BriefcaseBusiness,
  ChartScatter,
  Database,
  GraduationCap,
  Code2,
} from "lucide-react";

import { Swiper, SwiperSlide } from "swiper/react";
import { Autoplay, Pagination } from "swiper/modules";
import "swiper/css";
import "swiper/css/pagination";

const ExperienceCard = ({
  title,
  company,
  period,
  description,
  icon: Icon,
}) => (
  <div className="group relative overflow-hidden transform hover:-translate-y-2 transition-all duration-300">
    <div className="absolute inset-0 backdrop-blur-lg bg-white/5 rounded-lg" />
    <div className="absolute -inset-[2px] bg-gradient-to-r from-cyan-500 via-blue-500 to-purple-500 rounded-lg opacity-0 group-hover:opacity-100 animate-gradient-xy transition-all duration-500" />
    <div className="relative bg-gray-900/90 rounded-lg p-8 h-full border border-gray-800/50 shadow-xl backdrop-blur-xl">
      <div className="relative mb-6">
        <div className="absolute -inset-4 bg-gradient-to-r from-cyan-500 to-blue-500 opacity-25 rounded-full blur-xl group-hover:opacity-75 animate-pulse transition-all duration-500" />
        <Icon className="w-12 h-12 text-cyan-400 relative z-10 transform group-hover:rotate-12 transition-transform duration-300" />
      </div>
      <div className="space-y-3">
        <h3 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent font-mono">
          {title}
        </h3>
        <div className="flex justify-between items-center text-gray-300 font-mono">
          <span className="font-semibold text-blue-400">{company}</span>
          <span className="text-sm bg-blue-500/10 px-3 py-1 rounded-full">
            {period}
          </span>
        </div>
        <p className="text-gray-300 border-l-4 border-blue-500/50 pl-4 mt-4 leading-relaxed font-mono">
          {description}
        </p>
      </div>
      <div className="absolute top-4 right-4 w-20 h-20">
        <div className="absolute top-0 right-0 w-6 h-[2px] bg-cyan-500/50" />
        <div className="absolute top-0 right-0 w-[2px] h-6 bg-cyan-500/50" />
      </div>
      <div className="absolute bottom-4 left-4 w-20 h-20">
        <div className="absolute bottom-0 left-0 w-6 h-[2px] bg-purple-500/50" />
        <div className="absolute bottom-0 left-0 w-[2px] h-6 bg-purple-500/50" />
      </div>
    </div>
  </div>
);

const ExperienceSection = () => {
  const experiences = [
    {
      icon: Database,
      title: "Co-Founder & CEO",
      company: "DataTech LLC",
      period: "2025 - Present",
      description:
        "DataTech brings together a team of data engineers, AI/ML specialists, and business intelligence professionals to deliver end‑to‑end data-driven solutions and digital transformation services for Mongolian businesses and institutions.",
    },    
    {
      icon: Code2,
      title: "Founder",
      company: "Data School",
      period: "2018 - Present",
      description:
        "Founded Data School with a mission to close the education gap and empower the Mongolian community by providing high-quality, accessible content in Data Science, AI, and digital skills.",
    },
    {
      icon: Network,
      title: "Board Member",
      company: "Machine Learning Ulaanbaatar (MLUB)",
      period: "2020 - Present",
      description:
        "Machine Learning UB (MLUB) is a community dedicated to promoting AI and machine learning education, collaboration, and innovation in Mongolia.",
    },
    {
      icon: Cpu,
      title: "Data Science Manager",
      company: "Unitel Group",
      period: "2019 - 2023",
      description:
        "As a Data Science Manager at Mongolia's one of the largest Information and Communications Technology company, I managed a cross-functional team in designing and deploying advanced analytics solutions.",
    },
    {
      icon: Binary,
      title: "Senior Data Analyst",
      company: "Unitel Group",
      period: "2018 - 2019",
      description:
        "Led a team of data analysts across multiple projects, ensuring timely delivery of high-quality insights and models aligned with business objectives.",
    },
    {
      icon: BrainCircuit,
      title: "Research Specialist / Senior Business Analyst",
      company: "Unitel Group",
      period: "2012 - 2016",
      description:
        "Collaborated cross-functionally to deliver data-driven insights that supported strategic planning, operational improvements, and customer engagement initiatives.",
    },

  ];

  const volunteerExperiences = [
    {
      image: "/volunteer/kaggle-icon.png",
      role: "Co-organizer",
      org: "Kaggle Days Meetup Ulaanbaatar #2",
      date: "Sep 2021"
    },
    {
      image: "/volunteer/pieai.png",
      role: "Co-organizer",
      org: "Pie & AI Ulaanbaatar by DeepLearning.AI",
      date: "Mar 2021"
    },
    {
      image: "/volunteer/kaggle-icon.png",
      role: "Co-organizer",
      org: "Kaggle Days Meetup Ulaanbaatar #1",
      date: "Nov 2020"
    },
    {
      image: "/volunteer/mlub.png", 
      role: "Speaker & Organizer",
      org: "Data Nomads: AI & Data Science Conference",
      date: "Apr 2020"
    }
  ];

  return (
    <>
      {/* Experience Section */}
      <div className="min-h-screen bg-gradient-to-b relative overflow-hidden pt-32 pb-20">
        <div className="absolute inset-0 bg-[#04081A]" />
        <div className="absolute inset-0 bg-[linear-gradient(rgba(50,50,70,0.15)_1px,transparent_1px),linear-gradient(90deg,rgba(50,50,70,0.15)_1px,transparent_1px)] bg-[size:40px_40px] [mask-image:radial-gradient(ellipse_80%_80%_at_50%_50%,#000_70%,transparent_100%)]" />
        <div className="absolute inset-0">
          {[...Array(20)].map((_, i) => (
            <div
              key={i}
              className="absolute w-2 h-2 bg-blue-500/20 rounded-full animate-float"
              style={{
                top: `${Math.random() * 100}%`,
                left: `${Math.random() * 100}%`,
                animationDelay: `${Math.random() * 5}s`,
              }}
            />
          ))}
        </div>

        <div className="relative container mx-auto px-6 mt-10">
          <div className="flex flex-col items-center space-y-8 mb-20">
            <div className="relative">
              <h2 className="text-5xl md:text-7xl font-black text-transparent bg-gradient-to-r from-teal-400 to-blue-500 bg-clip-text text-center font-mono">
                Professional Journey
              </h2>
              <div className="absolute inset-0 -z-10 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 blur-3xl rounded-full" />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-10 max-w-7xl mx-auto">
            {experiences.map((exp, index) => (
              <ExperienceCard key={index} {...exp} />
            ))}
          </div>
        </div>

        <div className="absolute top-20 left-20 w-96 h-96 bg-cyan-500/10 rounded-full filter blur-3xl animate-pulse" />
        <div className="absolute bottom-20 right-20 w-96 h-96 bg-purple-500/10 rounded-full filter blur-3xl animate-pulse delay-1000" />
      </div>

      {/* Volunteer Experience Section */}
      <div className="relative bg-[#050818] py-20 px-6 overflow-hidden">
        <div className="absolute top-0 left-0 w-96 h-96 bg-emerald-400/10 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-0 right-0 w-96 h-96 bg-cyan-400/10 rounded-full blur-3xl animate-pulse delay-1000" />

        <div className="flex flex-col items-center mb-16 z-10 relative">
          <h2 className="text-4xl md:text-6xl font-extrabold text-transparent bg-gradient-to-r from-emerald-400 to-sky-500 bg-clip-text font-mono text-center">
            Volunteer Experience
          </h2>
          <p className="text-gray-400 mt-4 max-w-xl text-center font-mono">

          </p>
        </div>

        <Swiper
          modules={[Autoplay, Pagination]}
          autoplay={{ delay: 4000 }}
          loop
          pagination={{ clickable: true }}
          spaceBetween={30}
          breakpoints={{
            640: { slidesPerView: 1 },
            768: { slidesPerView: 3 },
            1024: { slidesPerView: 4 },
          }}
          className="z-10 relative"
        >
          {volunteerExperiences.map((item, index) => (
            <SwiperSlide key={index}>
              <div className="group relative rounded-2xl p-[2px] bg-transparent transition-all duration-300 hover:bg-gradient-to-r hover:from-cyan-400 hover:to-purple-500">
                <div className="bg-gray-900/80 backdrop-blur-lg border border-gray-800 rounded-2xl p-6 h-full shadow-lg transition-all duration-300 group-hover:border-transparent group-hover:bg-gray-900 flex flex-col items-center justify-center text-center">
                  
                  <div className="w-28 h-28 rounded-full border-4 border-cyan-500 mb-4 overflow-hidden flex items-center justify-center">
                    <img
                      src={item.image}
                      alt={item.org}
                      className="w-14 h-14 object-contain"
                    />
                  </div>
                  
                  <h3 className="text-lg font-bold text-white font-mono">{item.role}</h3>
                  <p className="text-cyan-400 font-mono">{item.org}</p>
                  <p className="text-sm text-gray-400 font-mono">{item.date}</p>
                </div>
              </div>
            </SwiperSlide>
          ))}
        </Swiper>
      </div>
    </>
  );
};

export default ExperienceSection;
