% twentysecondcv.cls — Version modular para Oscar Ayala
% Basado en la plantilla original de Carmine Spagnuolo
% Modificado para soporte de perfiles condicionales
\ProvidesClass{twentysecondcv}[2026/03/29 CV class - modular profiles]
\LoadClass{article}
\NeedsTeXFormat{LaTeX2e}

% ---------------------------
% Paquetes requeridos
% ---------------------------
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\RequirePackage{graphicx}
\RequirePackage{tikz}
\usetikzlibrary{calc}
\RequirePackage{xcolor}
\RequirePackage[absolute,overlay]{textpos}
\RequirePackage{ragged2e}
\RequirePackage{array}
\RequirePackage{etoolbox}
\RequirePackage{ifmtarg}
\RequirePackage{ifthen}
\RequirePackage{pgffor}
% marvosym removed — fontawesome5 provides all needed icons
\RequirePackage{parskip}
\RequirePackage{everypage}
\RequirePackage{hyperref}
\RequirePackage{fontawesome5}
\RequirePackage{enumitem}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax

% ---------------------------
% Flags de perfil (usar en .tex: \ProfileAtrue, etc.)
% ---------------------------
\newif\ifProfileA  % CAD/Manufactura
\newif\ifProfileB  % ML/Data
\newif\ifProfileC  % Mecatronica/IoT
\newif\ifProfileD  % Predictive Maintenance / Industrial AI (Cummins, manufactura serie)

% ---------------------------
% COLORES
% ---------------------------
\definecolor{white}{RGB}{255,255,255}
\definecolor{gray}{HTML}{4D4D4D}
\definecolor{sidecolor}{HTML}{E7E7E7}
\definecolor{mainblue}{HTML}{0E5484}
\definecolor{maingray}{HTML}{B9B9B9}

\renewcommand{\bfseries}{\color{gray}}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\sloppy

% ---------------------------
% GEOMETRIA
% ---------------------------
\newlength\SIDEBARWIDTH
\setlength\SIDEBARWIDTH{7cm}

\RequirePackage[left=\SIDEBARWIDTH,top=0.1cm,right=1cm,bottom=0.2cm,nohead,nofoot]{geometry}

\setlength{\TPHorizModule}{1cm}
\setlength{\TPVertModule}{1cm}
\newlength\imagewidth
\newlength\imagescale
\pgfmathsetlength{\imagewidth}{4.5cm}
\pgfmathsetlength{\imagescale}{\imagewidth/600}

% ---------------------------
% Titulo de seccion
% ---------------------------
\newlength{\TotalSectionLength}
\newlength{\SectionTitleLength}
\newcommand{\profilesection}[1]{%
  \setlength\TotalSectionLength{\linewidth}%
  \settowidth{\SectionTitleLength}{\huge #1 }%
  \addtolength\TotalSectionLength{-\SectionTitleLength}%
  \addtolength\TotalSectionLength{-2.22221pt}%
  \vspace{8pt}%
  {\color{black!80}\huge #1 \rule[0.15\baselineskip]{\TotalSectionLength}{1pt}}%
}

% ---------------------------
% Storage macros
% ---------------------------
\newcommand{\cvdatecontent}{}
\newcommand{\cvmailcontent}{}
\newcommand{\cvnumberphonecontent}{}
\newcommand{\cvaddresscontent}{}
\newcommand{\cvsitecontent}{}
\newcommand{\aboutmecontent}{}
\newcommand{\profilepiccontent}{}
\newcommand{\cvnamecontent}{}
\newcommand{\cvjobtitlecontent}{}
\newcommand{\cvlicensecontent}{}
\newcommand{\cvlinkedincontent}{}
\newcommand{\cvgithubcontent}{}

% Setters
\newcommand{\cvdate}[1]{\renewcommand{\cvdatecontent}{#1}}
\newcommand{\cvmail}[1]{\renewcommand{\cvmailcontent}{#1}}
\newcommand{\cvnumberphone}[1]{\renewcommand{\cvnumberphonecontent}{#1}}
\newcommand{\cvaddress}[1]{\renewcommand{\cvaddresscontent}{#1}}
\newcommand{\cvsite}[1]{\renewcommand{\cvsitecontent}{#1}}
\newcommand{\aboutme}[1]{\renewcommand{\aboutmecontent}{#1}}
\newcommand{\profilepic}[1]{\renewcommand{\profilepiccontent}{#1}}
\newcommand{\cvname}[1]{\renewcommand{\cvnamecontent}{#1}}
\newcommand{\cvjobtitle}[1]{\renewcommand{\cvjobtitlecontent}{#1}}
\newcommand{\cvlicense}[1]{\renewcommand{\cvlicensecontent}{#1}}
\newcommand{\cvlinkedin}[1]{\renewcommand{\cvlinkedincontent}{#1}}
\newcommand{\cvgithub}[1]{\renewcommand{\cvgithubcontent}{#1}}

% ---------------------------
% Iconos robustos con fallback
% ---------------------------
\DeclareRobustCommand{\FAcalendar}{\faCalendar}
\DeclareRobustCommand{\FAmap}{\faMapMarker}
\DeclareRobustCommand{\FAPhone}{\faPhone}
\DeclareRobustCommand{\FAGlobe}{\faGlobe}
\DeclareRobustCommand{\FAEnv}{\faEnvelope}
\DeclareRobustCommand{\FAcar}{\faCar}
\DeclareRobustCommand{\FAlinkedin}{\faLinkedin}
\DeclareRobustCommand{\FAgithub}{\faGithub}

\newcommand*\icon[1]{%
  \tikz[baseline=(char.base)]{%
    \node[shape=circle,draw,inner sep=0.9pt, minimum size=1mm, fill=mainblue, text=white] (char) {\scriptsize #1};%
  }%
}

% ---------------------------
% Skills (barras visuales por dominio)
% ---------------------------
\newcommand{\skills}{\relax}
\newcommand{\skillstext}{\relax}

\renewcommand{\skills}[1]{%
  \renewcommand{\skills}{%
    \begin{tikzpicture}
      \foreach [count=\i] \x/\y in {#1}{%
        \draw[fill=maingray,maingray] (0,\i) rectangle (6,\i+0.4);
        \draw[fill=white,mainblue] (0,\i) rectangle (\y,\i+0.4);
        \node [above right] at (0,\i+0.4) {\x};
      }
    \end{tikzpicture}%
  }%
}

\renewcommand{\skillstext}[1]{%
  \renewcommand{\skillstext}{%
    \begin{justify}
      \foreach [count=\i] \x/\y in {#1}{%
        \x\ \textbullet\ \y\par
      }
    \end{justify}%
  }%
}

% ---------------------------
% Sidebar background + linea divisoria
% ---------------------------
\definecolor{dividercolor}{HTML}{CFCFCF}
\newlength{\SideDividerWidth}
\setlength{\SideDividerWidth}{0.6pt}

\AddEverypageHook{%
  \begin{tikzpicture}[remember picture,overlay]
    \node[rectangle, fill=sidecolor, anchor=north west, minimum width=\SIDEBARWIDTH, minimum height=\paperheight]
      at (current page.north west) {};
    \draw[draw=dividercolor, line width=\SideDividerWidth]
      ($(current page.north west)+(\SIDEBARWIDTH,0)$) --
      ($(current page.north west)+(\SIDEBARWIDTH,-\paperheight)$);
  \end{tikzpicture}%
}

% ---------------------------
% makeprofile
% ---------------------------
\newcommand{\makeprofile}{%
  \begin{textblock}{6.3}(0.5,0.2)
    \ifthenelse{\equal{\profilepiccontent}{}}{}{%
      \begin{center}
        \begin{tikzpicture}[x=\imagescale,y=-\imagescale]
          \clip (600/2, 567/2) circle (567/2);
          \node[anchor=north west, inner sep=0pt, outer sep=0pt] at (0,0) {\includegraphics[width=\imagewidth]{\profilepiccontent}};
        \end{tikzpicture}
      \end{center}
    }

    {\Huge\color{mainblue}\cvnamecontent}

    {\Large\color{black!80}\cvjobtitlecontent}

    \renewcommand{\arraystretch}{1.05}
    \begin{tabular}{@{} >{\raggedright\arraybackslash}p{0.5cm} @{\hskip 0.2cm} >{\raggedright\arraybackslash}p{5.3cm} @{}}
      \ifthenelse{\equal{\cvaddresscontent}{}}{}{\icon{\FAmap} & \cvaddresscontent \\}
      \ifthenelse{\equal{\cvnumberphonecontent}{}}{}{\icon{\FAPhone} & \cvnumberphonecontent \\}
      \ifthenelse{\equal{\cvmailcontent}{}}{}{\icon{\FAEnv} & \href{mailto:\cvmailcontent}{\cvmailcontent} \\}
      \ifthenelse{\equal{\cvlicensecontent}{}}{}{\icon{\FAcar} & \cvlicensecontent \\}
      \ifthenelse{\equal{\cvlinkedincontent}{}}{}{\icon{\FAlinkedin} & \cvlinkedincontent \\}
      \ifthenelse{\equal{\cvgithubcontent}{}}{}{\icon{\FAgithub} & \cvgithubcontent \\}
    \end{tabular}

    \ifthenelse{\equal{\aboutmecontent}{}}{}{\profilesection{Perfil}\begin{justify}\aboutmecontent\end{justify}}

    \profilesection{Competencias}
    \skills
    \skillstext

  \end{textblock}%
}

% ---------------------------
% Seccion con iniciales decorativas
% ---------------------------
\newcommand*\round[2]{%
  \tikz[baseline=(char.base)]\node[anchor=north west, draw, rectangle, rounded corners, inner sep=1.6pt, minimum size=5.5mm, text height=3.6mm, fill=#2,#2,text=white](char){#1};%
}

\newcounter{colorCounter}
\newcommand{\sectioncolor}[1]{%
  {\round{#1}{\ifcase\value{colorCounter}maingray\or mainblue\or maingray\or mainblue\or maingray\or mainblue\or maingray\or mainblue\else maingray\fi}}%
  \stepcounter{colorCounter}%
}

\renewcommand{\section}[1]{{\color{gray}\Large\sectioncolor{#1}}}
\renewcommand{\subsection}[1]{\par\vspace{.5\parskip}{\large\color{gray} #1}\par\vspace{.25\parskip}}

% ---------------------------
% Entornos twenty
% ---------------------------
\newlength{\TwentyLeftColWidth}
\setlength{\TwentyLeftColWidth}{1.8cm}
\newlength{\TwentyRuleWidth}
\setlength{\TwentyRuleWidth}{0.8pt}
\newlength{\TwentyRuleGap}
\setlength{\TwentyRuleGap}{2mm}
\newlength{\TwentyColSep}
\setlength{\TwentyColSep}{\dimexpr 2\TwentyRuleGap + \TwentyRuleWidth\relax}

\setlength{\tabcolsep}{0pt}
\newenvironment{twenty}{\begin{list}{}{\setlength{\leftmargin}{0pt}\setlength{\labelsep}{0pt}\setlength{\itemsep}{\parsep}}}{\end{list}}

\newcommand{\twentyitem}[4]{%
  \item[]%
  \noindent
  \setbox0=\vbox{\hsize=\TwentyLeftColWidth\raggedright #1}%
  \setbox1=\vbox{\hsize=\dimexpr\linewidth-\TwentyLeftColWidth-\TwentyColSep\relax
     \textbf{#2}\hfill{\footnotesize #3}\par #4\par}%
  \noindent
  \begin{minipage}[t]{\TwentyLeftColWidth}\unvbox0\end{minipage}%
  \hspace{\TwentyRuleGap}%
  \begin{minipage}[t]{\TwentyRuleWidth}\vspace{0pt}\rule{\TwentyRuleWidth}{\dimexpr\ht1+\dp1\relax}\end{minipage}%
  \hspace{\TwentyRuleGap}%
  \begin{minipage}[t]{\dimexpr\linewidth-\TwentyLeftColWidth-\TwentyColSep\relax}\unvbox1\end{minipage}\par\vspace{1ex}%
}

\newenvironment{twentyshort}{\begin{list}{}{\setlength{\leftmargin}{0pt}\setlength{\labelsep}{0pt}}}{\end{list}}
\newcommand{\twentyitemshort}[2]{%
  \item[]
  \begin{minipage}[t]{\TwentyLeftColWidth}\raggedright #1\end{minipage}%
  \hspace{\TwentyColSep}%
  \begin{minipage}[t]{\dimexpr\linewidth-\TwentyLeftColWidth-\TwentyColSep\relax}\textbf{#2}\end{minipage}\par}

\endinput
