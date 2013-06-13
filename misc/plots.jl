

macro d3(ex::Expr)
	println(ex)
end

macro test(s)
    dump(string(s))
end

@test abcd
@test "ab cd"
@test <html><head> 54

fn = tempname()

io = open(fn, "w")

  a = :(  <!DOCTYPE html> )

    <html>
    <head>
        <script type=\"text/javascript\" src=\"http://mbostock.github.com/d3/d3.js\"></script>
    </head>
    <body>
        <div id=\"viz\"></div>
        <script type=\"text/javascript\">")
end

"' abc"d '"'

h = :(`<!DOCTYPE html>
<html>
<head>
    <script type="text/javascript" src="http://mbostock.github.com/d3/d3.js"></script>
</head>
<body>
    <div id="viz"></div>
    <script type="text/javascript">`)

h.exec

println(io, "<!DOCTYPE html>
<html>
<head>
    <script type=\"text/javascript\" src=\"http://mbostock.github.com/d3/d3.js\"></script>
</head>
<body>
    <div id=\"viz\"></div>
    <script type=\"text/javascript\">")

println(io, "var sampleSVG = d3.select(\"#viz\")
        .append(\"svg\")
        .attr(\"width\", 200)
        .attr(\"height\", 200);   ") 

println(io, "
    sampleSVG.append(\"circle\")
        .style(\"stroke\", \"red\")
        .style(\"fill\", \"blue\")
        .attr(\"r\", 60)
        .attr(\"cx\", 50)
        .attr(\"cy\", 50)
        .on(\"mouseover\", function(){d3.select(this).style(\"fill\", \"aliceblue\");})
        .on(\"mouseout\", function(){d3.select(this).style(\"fill\", \"white\");}); ")


println(io, "</script> </body> </html> ")
close(io)    

d3.open()

@d3 svg.selectAll("circle").
    data(data).
    attr("r", 2.5)
    
@d3 svg.selectAll("circle")
    .data(data)
    .attr("r", 2.5)



  .enter().append("circle")
    .attr("cx", function(d) { return d.x; })
    .attr("cy", function(d) { return d.y; })

:(4 | sin | cos)

Pkg.add("Calendar")
Pkg.add("JSON")


Pkg.update()
include("p:/apps/julia/packages/ICU/src/ICU.jl")

include("p:/apps/julia/packages/Dthree/src/dthree.jl")
using DThree
using JSON

plot(sin, 0, 3pi) | browse


f,a,b = sin, 0, pi
x = linspace(a, b, 250)
data = [{values=>[{:x=>x, :y=>f(x)} for x in x], :key=>"Sine"}] | JSON.to_json | I

d3 = D3()
q = D3Plot()

## cf. http://nvd3.org/ghpages/line.html


3/master/src/nv.d3.css n'a pas été chargée car son type MIME, « text/plain », n'est pas « text/css » @ file:///C:/TEMP/julia6.html


## var chart = nv.models.lineChart();
q * d3.var("chart").receiver("nv")._("models.lineChart")    

## chart.xAxis.axisLabel("x").tickFormat(d3.format(",.02"));
q * d3.receiver("chart.xAxis").axisLabel("x").tickFormat(D3().format(",.02f"))       

## chart.yAxis.axisLabel("y").tickFormat(d3.format(",.02"));
q * d3.receiver("chart.yAxis").axisLabel("y").tickFormat(D3().format(",.02f"))

##  d3.select("#chart svg").datum(data).transition().duration(500).call(chart);
q * d3.select("#chart svg").datum(data).transition().duration(500).call(I("chart"))

DThree.nv_addGraph(q) ## wrap in a function
browse(q)   


begin
    d3 = D3()
    q = D3Plot()
    q * d3.var("sample").attr("width", 200).attr("height", 200)
    q * I("sample").append("circle").style("stroke", "red").style("fill", "blue")
    q * I("sample").attr("r", 60).attr("cx", 50).attr("cy", 50)
end

        # .on("mouseover", function(){d3.select(this).style("fill", "aliceblue");})
        # .on("mouseout", function(){d3.select(this).style("fill", "white");});

q | browse

begin
    d3 = D3()
    q = D3Plot()
    q * d3.var("sample").attr("width", 200).attr("height", 200).
    append("circle").style("stroke", "red").style("fill", "blue").
    attr("r", 60).attr("cx", 50).attr("cy", 50)

    q | browse
end



dump(:(`ab"c <d >`))


stmt = 'ab"c <d >'



