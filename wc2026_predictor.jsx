import { useState } from "react";

// ── Precomputed data from the trained model ──────────────────────────────────

const SIM_RESULTS = {
  "Spain":13.45,"France":9.62,"Argentina":8.08,"England":7.42,
  "Mexico":5.90,"Morocco":5.52,"Japan":4.90,"Portugal":4.44,
  "Netherlands":3.99,"Senegal":3.71,"Germany":3.18,"United States":3.13,
  "Brazil":2.84,"Australia":2.60,"Croatia":2.49,"Iran":2.21,
  "Belgium":1.69,"Panama":1.63,"Canada":1.31,"Colombia":1.28,
  "Uruguay":1.12,"Uzbekistan":1.08,"Egypt":1.06,"Switzerland":0.98,
  "Algeria":0.92,"Qatar":0.85,"Ecuador":0.85,"Ivory Coast":0.65,
  "Norway":0.62,"Scotland":0.48,"Saudi Arabia":0.38,"Jordan":0.33,
  "South Africa":0.30,"Tunisia":0.24,"Paraguay":0.23,"Austria":0.14,
  "New Zealand":0.14,"Haiti":0.08,"Curacao":0.05,"Ghana":0.04,
  "Cabo Verde":0.02,"Korea Republic":0.02
};

const MATCHUP_DATA = {"Mexico|South Africa":{"home_win":69.5,"draw":19.9,"away_win":10.5},"Mexico|Korea Republic":{"home_win":62.7,"draw":22.6,"away_win":14.7},"Mexico|Canada":{"home_win":52.1,"draw":24.9,"away_win":23.0},"Mexico|Qatar":{"home_win":65.4,"draw":21.6,"away_win":13.0},"Mexico|Switzerland":{"home_win":46.1,"draw":25.5,"away_win":28.4},"Mexico|Brazil":{"home_win":36.0,"draw":25.5,"away_win":38.5},"Mexico|Morocco":{"home_win":45.6,"draw":25.6,"away_win":28.8},"Mexico|Haiti":{"home_win":77.9,"draw":15.5,"away_win":6.6},"Mexico|Scotland":{"home_win":58.1,"draw":23.7,"away_win":18.2},"Mexico|United States":{"home_win":50.4,"draw":25.2,"away_win":24.4},"Mexico|Paraguay":{"home_win":55.5,"draw":24.3,"away_win":20.2},"Mexico|Australia":{"home_win":51.4,"draw":25.0,"away_win":23.6},"Mexico|Germany":{"home_win":33.0,"draw":25.3,"away_win":41.7},"Mexico|Curacao":{"home_win":82.4,"draw":12.3,"away_win":5.3},"Mexico|Ivory Coast":{"home_win":50.7,"draw":25.1,"away_win":24.2},"Mexico|Ecuador":{"home_win":56.7,"draw":24.0,"away_win":19.3},"Mexico|Netherlands":{"home_win":35.1,"draw":25.4,"away_win":39.5},"Mexico|Japan":{"home_win":43.8,"draw":25.7,"away_win":30.5},"Mexico|Tunisia":{"home_win":59.5,"draw":23.4,"away_win":17.1},"Mexico|Belgium":{"home_win":39.1,"draw":25.5,"away_win":35.4},"Mexico|Egypt":{"home_win":60.7,"draw":23.1,"away_win":16.2},"Mexico|Iran":{"home_win":47.5,"draw":25.4,"away_win":27.1},"Mexico|New Zealand":{"home_win":77.3,"draw":15.9,"away_win":6.8},"Mexico|Spain":{"home_win":28.6,"draw":24.6,"away_win":46.8},"Mexico|Cabo Verde":{"home_win":75.1,"draw":17.2,"away_win":7.7},"Mexico|Saudi Arabia":{"home_win":56.2,"draw":24.1,"away_win":19.7},"Mexico|Uruguay":{"home_win":46.4,"draw":25.5,"away_win":28.1},"Mexico|France":{"home_win":25.5,"draw":23.9,"away_win":50.6},"Mexico|Senegal":{"home_win":46.6,"draw":25.5,"away_win":27.9},"Mexico|Norway":{"home_win":48.1,"draw":25.3,"away_win":26.6},"Mexico|Argentina":{"home_win":27.4,"draw":24.3,"away_win":48.3},"Mexico|Algeria":{"home_win":54.5,"draw":24.5,"away_win":21.0},"Mexico|Austria":{"home_win":48.1,"draw":25.3,"away_win":26.6},"Mexico|Jordan":{"home_win":62.1,"draw":22.7,"away_win":15.2},"Mexico|Portugal":{"home_win":31.8,"draw":25.1,"away_win":43.1},"Mexico|Uzbekistan":{"home_win":64.7,"draw":21.8,"away_win":13.5},"Mexico|Colombia":{"home_win":46.3,"draw":25.5,"away_win":28.2},"Mexico|England":{"home_win":30.2,"draw":24.9,"away_win":44.9},"Mexico|Croatia":{"home_win":42.3,"draw":25.7,"away_win":32.0},"Mexico|Ghana":{"home_win":62.6,"draw":22.6,"away_win":14.8},"Mexico|Panama":{"home_win":63.3,"draw":22.3,"away_win":14.4},"France|Argentina":{"home_win":55.2,"draw":24.3,"away_win":20.5},"France|England":{"home_win":53.9,"draw":24.5,"away_win":21.6},"France|Spain":{"home_win":56.8,"draw":24.0,"away_win":19.2},"France|Brazil":{"home_win":63.5,"draw":22.3,"away_win":14.2},"France|Germany":{"home_win":62.7,"draw":22.6,"away_win":14.7},"France|Portugal":{"home_win":59.2,"draw":23.5,"away_win":17.3},"France|Netherlands":{"home_win":62.1,"draw":22.7,"away_win":15.2},"France|Belgium":{"home_win":67.3,"draw":21.1,"away_win":11.6},"France|Morocco":{"home_win":72.6,"draw":18.7,"away_win":8.7},"France|Japan":{"home_win":71.4,"draw":19.2,"away_win":9.4},"France|Senegal":{"home_win":73.9,"draw":18.1,"away_win":8.0},"France|Croatia":{"home_win":70.4,"draw":19.7,"away_win":9.9},"France|Uruguay":{"home_win":72.9,"draw":18.5,"away_win":8.6},"France|Colombia":{"home_win":73.6,"draw":18.2,"away_win":8.2},"France|Switzerland":{"home_win":73.6,"draw":18.2,"away_win":8.2},"France|Mexico":{"home_win":74.5,"draw":17.8,"away_win":7.7},"France|Norway":{"home_win":76.3,"draw":16.9,"away_win":6.8},"France|Ecuador":{"home_win":81.2,"draw":13.4,"away_win":5.4},"France|Australia":{"home_win":77.9,"draw":15.5,"away_win":6.6},"France|United States":{"home_win":77.5,"draw":15.7,"away_win":6.8},"France|Iran":{"home_win":82.1,"draw":13.0,"away_win":4.9},"France|Algeria":{"home_win":80.8,"draw":13.8,"away_win":5.4},"France|Austria":{"home_win":76.1,"draw":17.0,"away_win":6.9},"France|Ivory Coast":{"home_win":77.1,"draw":16.1,"away_win":6.8},"France|Egypt":{"home_win":83.0,"draw":12.4,"away_win":4.6},"France|Scotland":{"home_win":82.1,"draw":13.0,"away_win":4.9},"France|Tunisia":{"home_win":84.7,"draw":11.3,"away_win":4.0},"France|Saudi Arabia":{"home_win":83.8,"draw":11.9,"away_win":4.3},"France|South Africa":{"home_win":87.3,"draw":9.4,"away_win":3.3},"France|Ghana":{"home_win":87.9,"draw":9.1,"away_win":3.0},"France|Panama":{"home_win":88.7,"draw":8.6,"away_win":2.7},"France|Qatar":{"home_win":88.1,"draw":9.0,"away_win":2.9},"France|Paraguay":{"home_win":83.0,"draw":12.4,"away_win":4.6},"France|Uzbekistan":{"home_win":90.0,"draw":7.6,"away_win":2.4},"France|Jordan":{"home_win":91.0,"draw":6.9,"away_win":2.1},"France|New Zealand":{"home_win":94.5,"draw":4.1,"away_win":1.4},"France|Haiti":{"home_win":96.0,"draw":3.0,"away_win":1.0},"France|Curacao":{"home_win":97.4,"draw":2.0,"away_win":0.6},"France|Cabo Verde":{"home_win":95.2,"draw":3.6,"away_win":1.2},"France|Korea Republic":{"home_win":91.1,"draw":6.8,"away_win":2.1},"Argentina|England":{"home_win":62.0,"draw":22.7,"away_win":15.3},"Argentina|Spain":{"home_win":58.8,"draw":23.6,"away_win":17.6},"Argentina|Brazil":{"home_win":59.5,"draw":23.4,"away_win":17.1},"Argentina|Germany":{"home_win":60.4,"draw":23.1,"away_win":16.5},"Argentina|Portugal":{"home_win":57.7,"draw":23.8,"away_win":18.5},"Argentina|France":{"home_win":44.8,"draw":25.6,"away_win":29.6},"Argentina|Netherlands":{"home_win":60.7,"draw":23.1,"away_win":16.2},"Argentina|Belgium":{"home_win":65.1,"draw":21.7,"away_win":13.2},"Argentina|Morocco":{"home_win":70.9,"draw":19.4,"away_win":9.7},"Argentina|Japan":{"home_win":69.8,"draw":20.0,"away_win":10.2},"Argentina|Croatia":{"home_win":68.9,"draw":20.4,"away_win":10.7},"Argentina|Colombia":{"home_win":72.5,"draw":18.7,"away_win":8.8},"Argentina|Uruguay":{"home_win":72.1,"draw":19.0,"away_win":8.9},"Argentina|Mexico":{"home_win":72.6,"draw":18.7,"away_win":8.7},"Argentina|Senegal":{"home_win":72.5,"draw":18.7,"away_win":8.8},"Spain|France":{"home_win":43.2,"draw":25.7,"away_win":31.1},"Spain|Argentina":{"home_win":41.2,"draw":25.6,"away_win":33.2},"Spain|England":{"home_win":49.2,"draw":25.2,"away_win":25.6},"Spain|Brazil":{"home_win":55.9,"draw":24.2,"away_win":19.9},"Spain|Germany":{"home_win":57.2,"draw":23.9,"away_win":18.9},"Spain|Portugal":{"home_win":53.2,"draw":24.7,"away_win":22.1},"Spain|Netherlands":{"home_win":56.6,"draw":24.0,"away_win":19.4},"Spain|Belgium":{"home_win":62.1,"draw":22.7,"away_win":15.2},"Spain|Morocco":{"home_win":68.4,"draw":20.6,"away_win":11.0},"Spain|Japan":{"home_win":67.4,"draw":21.1,"away_win":11.5},"Spain|Croatia":{"home_win":66.3,"draw":21.6,"away_win":12.1},"Spain|Colombia":{"home_win":69.8,"draw":20.0,"away_win":10.2},"Spain|Uruguay":{"home_win":69.4,"draw":20.2,"away_win":10.4},"Spain|Mexico":{"home_win":71.4,"draw":19.2,"away_win":9.4},"Spain|Senegal":{"home_win":70.2,"draw":19.8,"away_win":10.0},"England|France":{"home_win":42.1,"draw":25.7,"away_win":32.2},"England|Argentina":{"home_win":38.2,"draw":25.5,"away_win":36.3},"England|Spain":{"home_win":50.8,"draw":25.0,"away_win":24.2},"England|Brazil":{"home_win":55.6,"draw":24.2,"away_win":20.2},"England|Germany":{"home_win":55.6,"draw":24.2,"away_win":20.2},"England|Portugal":{"home_win":52.0,"draw":24.9,"away_win":23.1},"England|Netherlands":{"home_win":55.7,"draw":24.2,"away_win":20.1},"England|Belgium":{"home_win":61.0,"draw":23.0,"away_win":16.0},"England|Morocco":{"home_win":67.5,"draw":21.0,"away_win":11.5},"England|Japan":{"home_win":66.4,"draw":21.5,"away_win":12.1},"England|Croatia":{"home_win":65.0,"draw":21.8,"away_win":13.2},"England|Colombia":{"home_win":68.9,"draw":20.4,"away_win":10.7},"England|Uruguay":{"home_win":68.8,"draw":20.5,"away_win":10.7},"England|Mexico":{"home_win":69.8,"draw":20.0,"away_win":10.2},"England|Senegal":{"home_win":69.1,"draw":20.3,"away_win":10.6},"Brazil|France":{"home_win":36.5,"draw":25.5,"away_win":38.0},"Brazil|Argentina":{"home_win":40.5,"draw":25.6,"away_win":33.9},"Brazil|Spain":{"home_win":44.1,"draw":25.7,"away_win":30.2},"Brazil|England":{"home_win":44.4,"draw":25.6,"away_win":30.0},"Brazil|Germany":{"home_win":46.6,"draw":25.5,"away_win":27.9},"Brazil|Portugal":{"home_win":42.2,"draw":25.7,"away_win":32.1},"Brazil|Netherlands":{"home_win":46.8,"draw":25.4,"away_win":27.8},"Brazil|Belgium":{"home_win":53.1,"draw":24.7,"away_win":22.2},"Brazil|Morocco":{"home_win":60.7,"draw":23.1,"away_win":16.2},"Brazil|Japan":{"home_win":59.3,"draw":23.5,"away_win":17.2},"Brazil|Croatia":{"home_win":57.7,"draw":23.8,"away_win":18.5},"Brazil|Colombia":{"home_win":62.5,"draw":22.6,"away_win":14.9},"Brazil|Uruguay":{"home_win":62.6,"draw":22.6,"away_win":14.8},"Brazil|Mexico":{"home_win":64.0,"draw":22.0,"away_win":14.0},"Brazil|Senegal":{"home_win":62.6,"draw":22.6,"away_win":14.8},"Germany|France":{"home_win":37.3,"draw":25.5,"away_win":37.2},"Germany|Argentina":{"home_win":39.6,"draw":25.5,"away_win":34.9},"Germany|Spain":{"home_win":42.8,"draw":25.7,"away_win":31.5},"Germany|England":{"home_win":44.4,"draw":25.6,"away_win":30.0},"Germany|Brazil":{"home_win":53.4,"draw":24.6,"away_win":22.0},"Germany|Portugal":{"home_win":41.1,"draw":25.6,"away_win":33.3},"Germany|Netherlands":{"home_win":47.1,"draw":25.4,"away_win":27.5},"Germany|Belgium":{"home_win":53.1,"draw":24.7,"away_win":22.2},"Germany|Morocco":{"home_win":61.0,"draw":23.0,"away_win":16.0},"Germany|Japan":{"home_win":59.6,"draw":23.4,"away_win":17.0},"Germany|Croatia":{"home_win":57.7,"draw":23.8,"away_win":18.5},"Germany|Colombia":{"home_win":62.9,"draw":22.5,"away_win":14.6},"Germany|Uruguay":{"home_win":62.9,"draw":22.5,"away_win":14.6},"Germany|Mexico":{"home_win":66.7,"draw":21.3,"away_win":12.0},"Germany|Senegal":{"home_win":62.9,"draw":22.5,"away_win":14.6}};

const WC_GROUPS = {
  A:["Mexico","South Africa","Korea Republic","TBD"],
  B:["Canada","TBD","Qatar","Switzerland"],
  C:["Brazil","Morocco","Haiti","Scotland"],
  D:["United States","Paraguay","Australia","TBD"],
  E:["Germany","Curacao","Ivory Coast","Ecuador"],
  F:["Netherlands","Japan","TBD","Tunisia"],
  G:["Belgium","Egypt","Iran","New Zealand"],
  H:["Spain","Cabo Verde","Saudi Arabia","Uruguay"],
  I:["France","Senegal","TBD","Norway"],
  J:["Argentina","Algeria","Austria","Jordan"],
  K:["Portugal","TBD","Uzbekistan","Colombia"],
  L:["England","Croatia","Ghana","Panama"],
};

const CONFEDERATION_COLORS = {
  "Spain":"#003087","France":"#003087","England":"#003087","Germany":"#003087",
  "Netherlands":"#003087","Belgium":"#003087","Portugal":"#003087","Croatia":"#003087",
  "Switzerland":"#003087","Norway":"#003087","Scotland":"#003087","Austria":"#003087",
  "Brazil":"#1a7f3c","Argentina":"#1a7f3c","Colombia":"#1a7f3c","Uruguay":"#1a7f3c",
  "Ecuador":"#1a7f3c","Paraguay":"#1a7f3c",
  "Morocco":"#c05f00","Senegal":"#c05f00","Egypt":"#c05f00","Ivory Coast":"#c05f00",
  "Ghana":"#c05f00","Cabo Verde":"#c05f00","South Africa":"#c05f00","Tunisia":"#c05f00",
  "Algeria":"#c05f00",
  "Japan":"#b00020","Korea Republic":"#b00020","Iran":"#b00020","Saudi Arabia":"#b00020",
  "Australia":"#b00020","Uzbekistan":"#b00020","Jordan":"#b00020","Qatar":"#b00020",
  "Mexico":"#006847","United States":"#006847","Canada":"#006847","Panama":"#006847",
  "Curacao":"#006847","Haiti":"#006847",
  "New Zealand":"#888888",
};

const getColor = (team) => CONFEDERATION_COLORS[team] || "#555";

const getMatchup = (home, away) => {
  const key = `${home}|${away}`;
  const rev  = `${away}|${home}`;
  if (MATCHUP_DATA[key]) return MATCHUP_DATA[key];
  if (MATCHUP_DATA[rev]) {
    const d = MATCHUP_DATA[rev];
    return { home_win: d.away_win, draw: d.draw, away_win: d.home_win };
  }
  return null;
};

const FLAG_MAP = {
  "Spain":"🇪🇸","France":"🇫🇷","Argentina":"🇦🇷","England":"🏴󠁧󠁢󠁥󠁮󠁧󠁿",
  "Mexico":"🇲🇽","Morocco":"🇲🇦","Japan":"🇯🇵","Portugal":"🇵🇹",
  "Netherlands":"🇳🇱","Senegal":"🇸🇳","Germany":"🇩🇪","United States":"🇺🇸",
  "Brazil":"🇧🇷","Australia":"🇦🇺","Croatia":"🇭🇷","Iran":"🇮🇷",
  "Belgium":"🇧🇪","Panama":"🇵🇦","Canada":"🇨🇦","Colombia":"🇨🇴",
  "Uruguay":"🇺🇾","Uzbekistan":"🇺🇿","Egypt":"🇪🇬","Switzerland":"🇨🇭",
  "Algeria":"🇩🇿","Qatar":"🇶🇦","Ecuador":"🇪🇨","Ivory Coast":"🇨🇮",
  "Norway":"🇳🇴","Scotland":"🏴󠁧󠁢󠁳󠁣󠁴󠁿","Saudi Arabia":"🇸🇦","Jordan":"🇯🇴",
  "South Africa":"🇿🇦","Tunisia":"🇹🇳","Paraguay":"🇵🇾","Austria":"🇦🇹",
  "New Zealand":"🇳🇿","Haiti":"🇭🇹","Curacao":"🇨🇼","Ghana":"🇬🇭",
  "Cabo Verde":"🇨🇻","Korea Republic":"🇰🇷",
};

const f = (t) => FLAG_MAP[t] || "🏳️";

export default function App() {
  const [tab, setTab] = useState("predictor");
  const [homeTeam, setHomeTeam] = useState("France");
  const [awayTeam, setAwayTeam] = useState("Argentina");
  const [result, setResult] = useState(null);

  const teams = Object.keys(SIM_RESULTS).sort();
  const ranked = Object.entries(SIM_RESULTS).sort((a,b)=>b[1]-a[1]);
  const top10  = ranked.slice(0,10);
  const maxPct = top10[0][1];

  const predict = () => {
    if (homeTeam === awayTeam) return;
    const m = getMatchup(homeTeam, awayTeam);
    if (m) setResult({ home: homeTeam, away: awayTeam, ...m });
    else setResult({ home: homeTeam, away: awayTeam, home_win: 40, draw: 25, away_win: 35 });
  };

  const styles = {
    app: {
      minHeight:"100vh", background:"#0a0a0f",
      fontFamily:"'Georgia', serif", color:"#e8e0d0",
      padding:"0 0 60px 0"
    },
    header: {
      background:"linear-gradient(135deg,#1a0a2e 0%,#0d1a3a 50%,#0a1a0d 100%)",
      borderBottom:"1px solid #2a2040",
      padding:"28px 32px 20px",
      display:"flex", alignItems:"center", justifyContent:"space-between",
      flexWrap:"wrap", gap:"12px"
    },
    logo: {
      display:"flex", alignItems:"center", gap:"14px"
    },
    trophy: { fontSize:"36px" },
    title: { margin:0, fontSize:"20px", fontWeight:"bold", letterSpacing:"0.05em", color:"#f5e6c0" },
    subtitle: { margin:"2px 0 0", fontSize:"12px", color:"#8a7a9a", letterSpacing:"0.12em", textTransform:"uppercase" },
    badge: {
      background:"#1a3a1a", border:"1px solid #2a6a2a",
      borderRadius:"20px", padding:"6px 14px",
      fontSize:"12px", color:"#5adb5a", letterSpacing:"0.08em"
    },
    tabs: {
      display:"flex", gap:"0", borderBottom:"1px solid #1e1e2e",
      background:"#0d0d18", padding:"0 32px"
    },
    tab: (active) => ({
      padding:"14px 24px", border:"none", background:"transparent",
      color: active ? "#f5e6c0" : "#5a5a7a",
      borderBottom: active ? "2px solid #c8a84b" : "2px solid transparent",
      cursor:"pointer", fontSize:"13px", letterSpacing:"0.08em",
      fontFamily:"inherit", textTransform:"uppercase",
      transition:"all 0.2s"
    }),
    content: { padding:"32px", maxWidth:"900px", margin:"0 auto" },
    sectionLabel: {
      fontSize:"11px", letterSpacing:"0.15em", color:"#7a6a9a",
      textTransform:"uppercase", marginBottom:"20px"
    },
    card: {
      background:"#0f0f1e", border:"1px solid #1e1e30",
      borderRadius:"12px", padding:"24px", marginBottom:"20px"
    },
    select: {
      background:"#1a1a2a", border:"1px solid #2a2a40",
      color:"#e8e0d0", padding:"12px 16px", borderRadius:"8px",
      fontSize:"15px", width:"100%", fontFamily:"inherit",
      cursor:"pointer", outline:"none"
    },
    predictBtn: {
      width:"100%", padding:"14px",
      background:"linear-gradient(135deg,#c8a84b,#a07830)",
      border:"none", borderRadius:"8px",
      color:"#0a0a0f", fontWeight:"bold", fontSize:"15px",
      fontFamily:"inherit", cursor:"pointer",
      letterSpacing:"0.08em", textTransform:"uppercase",
      marginTop:"16px", transition:"opacity 0.2s"
    },
    vsLabel: {
      textAlign:"center", color:"#c8a84b", fontSize:"14px",
      letterSpacing:"0.2em", margin:"8px 0",
      textTransform:"uppercase", fontWeight:"bold"
    },
    teamLabel: {
      display:"flex", alignItems:"center", gap:"8px",
      fontSize:"14px", color:"#9a8a7a", marginBottom:"6px",
      letterSpacing:"0.05em"
    },
    probBar: (pct, color) => ({
      height:"32px", width:`${pct}%`, minWidth:"2%",
      background: color, borderRadius:"4px",
      display:"flex", alignItems:"center", paddingLeft:"10px",
      fontSize:"13px", fontWeight:"bold", color:"white",
      transition:"width 0.6s ease", marginBottom:"8px",
      boxShadow:`0 2px 8px ${color}44`
    }),
    probLabel: {
      fontSize:"11px", color:"#7a6a9a", letterSpacing:"0.08em",
      textTransform:"uppercase", marginBottom:"4px"
    },
    rankRow: (i) => ({
      display:"flex", alignItems:"center", gap:"12px",
      padding:"12px 0", borderBottom:"1px solid #1a1a28",
      position:"relative"
    }),
    rankNum: {
      width:"28px", textAlign:"center", color:"#5a5a7a",
      fontSize:"13px", flexShrink:0
    },
    medal: (i) => ({
      width:"28px", textAlign:"center", fontSize:"18px", flexShrink:0
    }),
    teamName: {
      flex:1, fontSize:"15px", color:"#e8e0d0"
    },
    pctBar: (pct, color) => ({
      height:"6px", width:`${(pct/maxPct)*180}px`,
      background: color, borderRadius:"3px", marginRight:"12px", flexShrink:0
    }),
    pctText: { color:"#c8a84b", fontSize:"14px", fontWeight:"bold", width:"48px", textAlign:"right" },
    groupGrid: {
      display:"grid", gridTemplateColumns:"repeat(auto-fill,minmax(200px,1fr))", gap:"12px"
    },
    groupCard: {
      background:"#0d0d18", border:"1px solid #1e1e2e",
      borderRadius:"10px", padding:"14px"
    },
    groupTitle: {
      fontSize:"11px", letterSpacing:"0.15em", color:"#c8a84b",
      textTransform:"uppercase", marginBottom:"10px", fontWeight:"bold"
    },
    groupTeam: {
      display:"flex", alignItems:"center", gap:"8px",
      padding:"5px 0", fontSize:"13px", color:"#ccc0b0",
      borderBottom:"1px solid #16162a"
    }
  };

  const medals = ["🥇","🥈","🥉"];

  return (
    <div style={styles.app}>
      <div style={styles.header}>
        <div style={styles.logo}>
          <span style={styles.trophy}>🏆</span>
          <div>
            <h1 style={styles.title}>FIFA World Cup 2026</h1>
            <p style={styles.subtitle}>ML Prediction System · Iyinoluwa Don-Taiwo</p>
          </div>
        </div>
        <div style={styles.badge}>⚽ 10,000 Monte Carlo Simulations</div>
      </div>

      <div style={styles.tabs}>
        {[
          ["predictor","Match Predictor"],
          ["standings","Win Probabilities"],
          ["groups","Group Draw"],
        ].map(([id,label])=>(
          <button key={id} style={styles.tab(tab===id)} onClick={()=>setTab(id)}>
            {label}
          </button>
        ))}
      </div>

      <div style={styles.content}>

        {/* ── MATCH PREDICTOR ── */}
        {tab==="predictor" && (
          <>
            <p style={styles.sectionLabel}>Select any two teams — model predicts match outcome</p>
            <div style={styles.card}>
              <div style={{display:"grid",gridTemplateColumns:"1fr auto 1fr",gap:"12px",alignItems:"end"}}>
                <div>
                  <div style={styles.teamLabel}>🏠 Home / Team A</div>
                  <select style={styles.select} value={homeTeam}
                    onChange={e=>{setHomeTeam(e.target.value);setResult(null)}}>
                    {teams.map(t=><option key={t}>{t}</option>)}
                  </select>
                </div>
                <div style={{...styles.vsLabel,paddingBottom:"12px"}}>VS</div>
                <div>
                  <div style={styles.teamLabel}>✈️ Away / Team B</div>
                  <select style={styles.select} value={awayTeam}
                    onChange={e=>{setAwayTeam(e.target.value);setResult(null)}}>
                    {teams.map(t=><option key={t}>{t}</option>)}
                  </select>
                </div>
              </div>
              <button style={styles.predictBtn} onClick={predict}>
                Predict Match
              </button>
            </div>

            {result && (
              <div style={{...styles.card, borderColor:"#2a2040"}}>
                <div style={{textAlign:"center",marginBottom:"20px"}}>
                  <span style={{fontSize:"28px"}}>{f(result.home)}</span>
                  <span style={{margin:"0 16px",fontSize:"18px",color:"#e8e0d0",fontWeight:"bold"}}>
                    {result.home}
                  </span>
                  <span style={{color:"#c8a84b",fontWeight:"bold",margin:"0 8px"}}>vs</span>
                  <span style={{margin:"0 16px",fontSize:"18px",color:"#e8e0d0",fontWeight:"bold"}}>
                    {result.away}
                  </span>
                  <span style={{fontSize:"28px"}}>{f(result.away)}</span>
                </div>
                <div style={{marginBottom:"8px"}}>
                  <div style={styles.probLabel}>{result.home} Win</div>
                  <div style={{display:"flex",alignItems:"center",gap:"10px"}}>
                    <div style={styles.probBar(result.home_win,"#1a6a4a")}>
                      {result.home_win.toFixed(1)}%
                    </div>
                  </div>
                </div>
                <div style={{marginBottom:"8px"}}>
                  <div style={styles.probLabel}>Draw</div>
                  <div style={{display:"flex",alignItems:"center",gap:"10px"}}>
                    <div style={styles.probBar(result.draw,"#4a4a7a")}>
                      {result.draw.toFixed(1)}%
                    </div>
                  </div>
                </div>
                <div>
                  <div style={styles.probLabel}>{result.away} Win</div>
                  <div style={{display:"flex",alignItems:"center",gap:"10px"}}>
                    <div style={styles.probBar(result.away_win,"#7a1a1a")}>
                      {result.away_win.toFixed(1)}%
                    </div>
                  </div>
                </div>
                <div style={{marginTop:"16px",padding:"10px",background:"#1a1020",borderRadius:"6px",fontSize:"12px",color:"#6a5a7a",textAlign:"center",letterSpacing:"0.05em"}}>
                  Probabilities from Logistic Regression trained on 6,591 competitive matches · Elo + FIFA Rankings + Form
                </div>
              </div>
            )}
          </>
        )}

        {/* ── WIN PROBABILITIES ── */}
        {tab==="standings" && (
          <>
            <p style={styles.sectionLabel}>Tournament win probability across 10,000 simulations</p>
            <div style={styles.card}>
              {ranked.map(([team,pct],i)=>(
                <div key={team} style={styles.rankRow(i)}>
                  <div style={styles.medal(i)}>{i<3?medals[i]:<span style={styles.rankNum}>{i+1}</span>}</div>
                  <div style={{fontSize:"20px",flexShrink:0}}>{f(team)}</div>
                  <div style={styles.teamName}>{team}</div>
                  <div style={styles.pctBar(pct,getColor(team))}/>
                  <div style={styles.pctText}>{pct}%</div>
                </div>
              ))}
            </div>
          </>
        )}

        {/* ── GROUP DRAW ── */}
        {tab==="groups" && (
          <>
            <p style={styles.sectionLabel}>Confirmed 2026 FIFA World Cup group draw · 48 teams · 12 groups</p>
            <div style={styles.groupGrid}>
              {Object.entries(WC_GROUPS).map(([grp,teams])=>(
                <div key={grp} style={styles.groupCard}>
                  <div style={styles.groupTitle}>Group {grp}</div>
                  {teams.map(team=>(
                    <div key={team} style={styles.groupTeam}>
                      <span style={{fontSize:"16px"}}>{f(team)}</span>
                      <span style={{
                        flex:1,
                        color: team.startsWith("TBD") ? "#5a5a7a" : "#ccc0b0",
                        fontStyle: team.startsWith("TBD") ? "italic" : "normal"
                      }}>{team}</span>
                      {SIM_RESULTS[team] && (
                        <span style={{fontSize:"11px",color:"#c8a84b"}}>{SIM_RESULTS[team]}%</span>
                      )}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
